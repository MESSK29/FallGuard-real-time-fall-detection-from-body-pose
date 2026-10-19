"""main.py - FastAPI Backend Service for FallGuard.

Key Features:
- GET /health -> {"status":"ok"}
- WebSocket /ws -> Ingests and validates 33 MediaPipe landmarks [x,y,z,visibility]
  and evaluates real-time FallDetector ML state machine.
- GET /alerts -> Returns in-memory list of confirmed alerts
- CORS enabled for local development
- Environment configuration via python-dotenv
- Serves app/frontend as static files at root "/"
- Detailed connection logging
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tempfile

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.backend.detector import FallDetector, load_model_bundle
from app.backend.incident_clip import (
    clip_processing_semaphore,
    is_ffmpeg_available,
    process_incident_clip,
    send_telegram_video,
)

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fallguard.backend")

# ---------------------------------------------------------------------------
# Load Configuration from .env
# ---------------------------------------------------------------------------
load_dotenv()

PORT = int(os.getenv("PORT", 8000))
MODEL_NAME = os.getenv("MODEL_NAME", "xgb")
PROBABILITY_THRESHOLD_ENV = os.getenv("PROBABILITY_THRESHOLD")
PROBABILITY_THRESHOLD = float(PROBABILITY_THRESHOLD_ENV) if PROBABILITY_THRESHOLD_ENV else None
LYING_SECONDS = float(os.getenv("LYING_SECONDS", 2.0))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TIMEZONE_STR = os.getenv("TIMEZONE", "UTC").strip()

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logger.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing. Telegram alerts are disabled.")

# Pre-load shared model bundle once at startup
try:
    load_model_bundle(MODEL_NAME)
    logger.info(f"Model bundle for '{MODEL_NAME}' successfully pre-loaded.")
except Exception as e:
    logger.warning(f"Could not pre-load model bundle for '{MODEL_NAME}': {e}")

logger.info(
    f"Configuration loaded: PORT={PORT}, MODEL_NAME={MODEL_NAME}, "
    f"PROBABILITY_THRESHOLD={PROBABILITY_THRESHOLD}, LYING_SECONDS={LYING_SECONDS}, "
    f"TIMEZONE={TIMEZONE_STR}"
)

# ---------------------------------------------------------------------------
# Application Initialization & Middleware
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FallGuard API",
    description="Privacy-Preserving Camera-Based Fall Detection Backend",
    version="0.3.0",
)

# Enable CORS for local development across origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory alerts registry and active connection tracking
in_memory_alerts: List[Dict[str, Any]] = []
active_detectors: List[FallDetector] = []

# Rate limiting state: client_id -> timestamps
client_real_alert_timestamps: Dict[str, float] = {}
client_test_alert_timestamps: Dict[str, List[float]] = {}
client_error_alert_timestamps: Dict[str, float] = {}
client_clip_timestamps: Dict[str, List[float]] = {}
client_test_clip_timestamps: Dict[str, List[float]] = {}

# Consent state: single source of truth dict client_id -> bool
clip_consent: Dict[str, bool] = {}
client_clip_consents = clip_consent  # backward-compatible reference

# Verified Telegram chat IDs mapping: client_id -> verified_chat_id
client_verified_chat_ids: Dict[str, str] = {}
active_verified_chat_id: Optional[str] = None


def check_clip_rate_limit(client_id: str) -> bool:
    """Enforce at most 5 incident clip uploads per hour (3600 s) per client_id."""
    now = time.time()
    one_hour_ago = now - 3600.0
    ts_list = [t for t in client_clip_timestamps.get(client_id, []) if t > one_hour_ago]
    if len(ts_list) >= 5:
        return False
    ts_list.append(now)
    client_clip_timestamps[client_id] = ts_list
    return True


def check_test_clip_rate_limit(client_id: str) -> bool:
    """Enforce at most 2 test clip requests per hour (3600 s) per client_id."""
    now = time.time()
    one_hour_ago = now - 3600.0
    ts_list = [t for t in client_test_clip_timestamps.get(client_id, []) if t > one_hour_ago]
    if len(ts_list) >= 2:
        return False
    ts_list.append(now)
    client_test_clip_timestamps[client_id] = ts_list
    return True


def get_target_chat_id(client_id: Optional[str] = None, explicit_chat: Optional[str] = None) -> Optional[str]:
    """Resolve target Telegram chat ID with priority: explicit -> client verified -> active verified -> env fallback."""
    if explicit_chat:
        validated = validate_chat_id(explicit_chat)
        if validated:
            return validated
    if client_id and client_id in client_verified_chat_ids:
        return client_verified_chat_ids[client_id]
    if active_verified_chat_id:
        return active_verified_chat_id
    env_chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    return env_chat if env_chat else None


def check_error_alert_rate_limit(client_id: str) -> bool:
    """Enforce at most 1 error notification per 30s per client to prevent spamming."""
    now = time.time()
    last_t = client_error_alert_timestamps.get(client_id, 0.0)
    if now - last_t < 30.0:
        return False
    client_error_alert_timestamps[client_id] = now
    return True


def get_client_id(request: Optional[Request] = None) -> str:
    """Extract client UUID from header X-Client-Id or query parameter."""
    if request is None:
        return "anonymous"
    cid = request.headers.get("X-Client-Id") or request.query_params.get("client_id")
    if cid and isinstance(cid, str) and cid.strip():
        return cid.strip()
    return "anonymous"


def validate_chat_id(raw_id: Any) -> Optional[str]:
    """Validate that chat ID matches numeric pattern (optionally starting with minus)."""
    if raw_id is None:
        return None
    s = str(raw_id).strip()
    if not s:
        return None
    if re.match(r"^-?\d+$", s):
        return s
    return None


def mask_chat_id(chat_id: Optional[str]) -> str:
    """Return masked chat ID showing last 4 digits only (e.g., '...1585')."""
    if not chat_id:
        return ""
    if len(chat_id) <= 4:
        return f"...{chat_id}"
    return f"...{chat_id[-4:]}"


def check_real_alert_rate_limit(client_id: str) -> bool:
    """Enforce at most 1 real alert message per 60 s per client_id."""
    now = time.time()
    last_t = client_real_alert_timestamps.get(client_id, 0.0)
    if now - last_t < 60.0:
        return False
    client_real_alert_timestamps[client_id] = now
    return True


def check_test_alert_rate_limit(client_id: str) -> bool:
    """Enforce at most 3 test messages per hour (3600 s) per client_id."""
    now = time.time()
    recent = [t for t in client_test_alert_timestamps.get(client_id, []) if now - t < 3600.0]
    if len(recent) >= 3:
        return False
    recent.append(now)
    client_test_alert_timestamps[client_id] = recent
    return True


def get_local_time_str(dt: Optional[datetime] = None) -> str:
    """Format local datetime according to TIMEZONE in format '17 Oct 2026, 6:22 PM'."""
    tz_env = os.getenv("TIMEZONE", "UTC").strip()
    try:
        tz = ZoneInfo(tz_env)
    except Exception:
        tz = timezone.utc

    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    dt_local = dt.astimezone(tz)
    hour_12 = int(dt_local.strftime("%I"))
    return f"{dt_local.day} {dt_local.strftime('%b')} {dt_local.year}, {hour_12}:{dt_local.strftime('%M %p')}"


def _sanitize_log(msg: str, token: str) -> str:
    """Sanitize any occurrence of the Telegram bot token from log messages."""
    if token and token in msg:
        return msg.replace(token, "[REDACTED]")
    return msg


def _sync_send_telegram(message: str, chat_id: Optional[str] = None) -> bool:
    """Synchronous worker to send a message via Telegram Bot API with 1 retry."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    target_chat = chat_id or os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not target_chat:
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": target_chat, "text": message}

    for attempt in range(2):
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200 and resp.json().get("ok"):
                logger.info(f"Telegram notification sent successfully to {mask_chat_id(target_chat)}.")
                return True
            else:
                sanitized_body = _sanitize_log(resp.text, token)
                logger.warning(f"Telegram API responded with code {resp.status_code}: {sanitized_body}")
        except Exception as exc:
            sanitized_err = _sanitize_log(str(exc), token)
            logger.warning(f"Telegram send attempt {attempt + 1} failed: {sanitized_err}")

        if attempt == 0:
            time.sleep(0.5)

    return False


async def send_telegram(message: str, chat_id: Optional[str] = None) -> bool:
    """Asynchronously send Telegram notification in a thread with a 10s timeout."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    target_chat = chat_id or os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not target_chat:
        return False

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_sync_send_telegram, message, target_chat),
            timeout=10.0,
        )
    except asyncio.TimeoutError:
        logger.warning("Telegram request timed out after 10s.")
        return False
    except Exception as exc:
        sanitized_err = _sanitize_log(str(exc), token)
        logger.warning(f"Error executing send_telegram: {sanitized_err}")
        return False


def _sync_verify_telegram(chat_id: str, custom_token: Optional[str] = None) -> Dict[str, Any]:
    """Perform real internal checks against Telegram Bot API for bot token and target chat ID."""
    token = (custom_token or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
    if not token:
        return {
            "ok": False,
            "valid": False,
            "error": "Telegram Bot Token is missing in configuration (.env).",
            "bot_username": None,
        }

    # 1. Internal check: Verify Bot Token via Telegram API getMe
    try:
        me_resp = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
        if me_resp.status_code != 200 or not me_resp.json().get("ok"):
            err_desc = "Invalid or unauthorized bot token"
            try:
                err_desc = me_resp.json().get("description", err_desc)
            except Exception:
                pass
            return {
                "ok": False,
                "valid": False,
                "error": f"Bot token check failed ({me_resp.status_code}): {err_desc}",
                "bot_username": None,
            }
        bot_data = me_resp.json().get("result", {})
        bot_username = bot_data.get("username", "")
        bot_name = bot_data.get("first_name", "FallGuard Bot")
    except Exception as exc:
        sanitized_err = _sanitize_log(str(exc), token)
        return {
            "ok": False,
            "valid": False,
            "error": f"Could not connect to Telegram API: {sanitized_err}",
            "bot_username": None,
        }

    # 2. Internal check: Format validation of chat_id
    valid_chat = validate_chat_id(chat_id)
    if not valid_chat:
        return {
            "ok": False,
            "valid": False,
            "error": "Invalid Chat ID format. Must be numeric digits (e.g. 5513741585).",
            "bot_username": bot_username,
            "bot_name": bot_name,
        }

    # 3. Internal check: Send live verification message to this specific chat ID
    now_str = get_local_time_str()
    bot_handle_str = f"@{bot_username}" if bot_username else "FallGuard Bot"
    verify_msg = (
        f"🛡️ FallGuard Connection Verified\n\n"
        f"✅ Bot: {bot_handle_str} ({bot_name})\n"
        f"💬 Chat ID: {valid_chat}\n"
        f"⏰ Verified: {now_str}\n\n"
        f"Telegram alerts are now linked to this chat. Emergency fall alerts and system notifications will be sent to this chat only."
    )
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": valid_chat, "text": verify_msg}

    try:
        send_resp = requests.post(url, json=payload, timeout=10)
        if send_resp.status_code == 200 and send_resp.json().get("ok"):
            logger.info(f"Telegram verification successful for chat {mask_chat_id(valid_chat)} via {bot_handle_str}.")
            return {
                "ok": True,
                "valid": True,
                "chat_id": valid_chat,
                "masked_chat_id": mask_chat_id(valid_chat),
                "bot_username": bot_username,
                "bot_name": bot_name,
                "message": f"Token is valid and verification message sent to chat {mask_chat_id(valid_chat)} via {bot_handle_str}.",
            }
        else:
            desc = "Unknown Telegram API error"
            try:
                desc = send_resp.json().get("description", desc)
            except Exception:
                pass

            help_hint = ""
            desc_lower = desc.lower()
            if "chat not found" in desc_lower:
                help_hint = f" The caregiver must open {bot_handle_str} in Telegram and click 'Start' first."
            elif "blocked" in desc_lower:
                help_hint = f" The bot {bot_handle_str} is blocked by this user in Telegram."

            return {
                "ok": False,
                "valid": False,
                "error": f"Telegram check failed ({send_resp.status_code}): {desc}.{help_hint}",
                "bot_username": bot_username,
                "bot_name": bot_name,
            }
    except Exception as exc:
        sanitized_err = _sanitize_log(str(exc), token)
        return {
            "ok": False,
            "valid": False,
            "error": f"Failed to send verification message to Telegram: {sanitized_err}",
            "bot_username": bot_username,
            "bot_name": bot_name,
        }


async def send_system_error_telegram(
    error_message: str,
    error_type: str = "System",
    client_id: str = "anonymous",
    chat_id: Optional[str] = None,
) -> bool:
    """Sends a system error notification to the client's verified Telegram chat ID."""
    if not check_error_alert_rate_limit(client_id):
        logger.warning(f"Rate limit: error alert suppressed for client {client_id} (max 1 per 30s).")
        return False

    target_chat = get_target_chat_id(client_id=client_id, explicit_chat=chat_id)
    if not target_chat:
        logger.info(f"No Telegram chat configured/verified for client {client_id}; skipping error alert.")
        return False

    now_local = get_local_time_str()
    msg = (
        f"⚠️ FallGuard System Error Alert\n\n"
        f"🔧 Type: {error_type}\n"
        f"⏰ Time: {now_local}\n"
        f"❌ Error: {error_message}\n\n"
        f"Please check the FallGuard monitoring device."
    )
    return await send_telegram(msg, chat_id=target_chat)


def create_alert(
    probability: float = 1.0,
    client_id: str = "anonymous",
    is_test: bool = False,
    status: Optional[str] = None,
    destination_chat_id: Optional[str] = None,
    clip_status: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Helper to register a new alert in in-memory storage."""
    prob = kwargs.get("peak_prob", kwargs.get("prob", probability))
    now_utc = datetime.now(timezone.utc)
    alert_id = len(in_memory_alerts) + 1
    local_time = get_local_time_str(now_utc)
    stat = status or ("test" if is_test else "confirmed")
    dest_chat = destination_chat_id or get_target_chat_id(client_id=client_id)
    has_clip_consent = client_clip_consents.get(client_id, False)
    default_clip_status = clip_status or ("pending" if has_clip_consent else "disabled")

    record: Dict[str, Any] = {
        "id": alert_id,
        "client_id": client_id,
        "time": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "local_time": local_time,
        "probability": round(float(prob), 4),
        "peak_probability": round(float(prob), 4),
        "status": stat,
        "telegram_sent": False,
        "destination_chat_id": dest_chat or "",
        "clip_status": default_clip_status,
    }
    if is_test:
        record["is_test"] = True
    if kwargs.get("is_test_clip"):
        record["is_test_clip"] = True
    in_memory_alerts.append(record)
    return record


async def dispatch_fall_telegram(record: Dict[str, Any], chat_id: Optional[str] = None) -> None:
    """Send Telegram message for confirmed fall alert and record delivery status."""
    client_id = record.get("client_id", "anonymous")
    target_chat = chat_id or record.get("destination_chat_id") or get_target_chat_id(client_id=client_id)
    record["destination_chat_id"] = target_chat or ""

    if not check_real_alert_rate_limit(client_id):
        logger.warning(f"Rate limit: real alert Telegram suppressed for client {client_id} (max 1 per 60s).")
        record["telegram_sent"] = False
        return

    has_clip_consent = client_clip_consents.get(client_id, False)
    clip_notice = " A video clip will follow in about 10 s." if has_clip_consent else ""
    msg = f"FALL DETECTED at {record['local_time']}. Please check on them.{clip_notice}"
    sent = await send_telegram(msg, chat_id=target_chat)
    record["telegram_sent"] = sent
    if not has_clip_consent:
        record["clip_status"] = "disabled"


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Health check endpoint required by project spec."""
    return {"status": "ok"}


@app.get("/api/health")
async def api_health():
    """Diagnostic health check with system metadata."""
    return {
        "status": "ok",
        "system": "FallGuard",
        "model_name": MODEL_NAME,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "lying_seconds": LYING_SECONDS,
        "incident_clips_supported": is_ffmpeg_available(),
        "disclaimer": "Prototype for educational/research purposes only. Not a certified medical device.",
    }


@app.get("/features")
async def get_features():
    """Report platform feature availability without exposing secrets."""
    ffmpeg_ok = is_ffmpeg_available()
    return {
        "incident_clips_supported": ffmpeg_ok,
        "ffmpeg_available": ffmpeg_ok,
        "video_recording": ffmpeg_ok,
        "system": "FallGuard",
    }


@app.get("/alerts")
async def get_alerts(request: Request = None):
    """Returns in-memory list of verified fall alerts for the calling client_id (newest first)."""
    cid = get_client_id(request) if request is not None else None
    alerts_list = reversed(in_memory_alerts) if cid is None else [a for a in reversed(in_memory_alerts) if a.get("client_id") == cid]
    client_alerts = []
    for a in alerts_list:
        record = dict(a)
        # Never expose full chat ID; mask to last 4 digits
        if "destination_chat_id" in record:
            record["destination_chat_id"] = mask_chat_id(record["destination_chat_id"])
        record["chat_id_masked"] = mask_chat_id(a.get("destination_chat_id"))
        client_alerts.append(record)
    return client_alerts


@app.post("/alerts/clear")
async def clear_alerts(request: Request = None):
    """Clears alert history for the calling client_id only."""
    global in_memory_alerts
    cid = get_client_id(request)
    initial_count = len(in_memory_alerts)
    in_memory_alerts = [a for a in in_memory_alerts if a.get("client_id") != cid]
    cleared = initial_count - len(in_memory_alerts)
    return {"status": "ok", "cleared_count": cleared}


@app.post("/alerts/{alert_id}/dismiss")
async def dismiss_alert(alert_id: int, request: Request = None):
    """Sets status to dismissed, sends Telegram notification, and returns updated alert."""
    cid = get_client_id(request) if request is not None else None
    target_alert = None
    for alert in in_memory_alerts:
        if alert["id"] == alert_id:
            if cid is None or alert.get("client_id") == cid:
                target_alert = alert
                break

    if target_alert is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

    target_alert["status"] = "dismissed"
    if target_alert.get("clip_status") in ("pending", "processing"):
        target_alert["clip_status"] = "skipped_dismissed"

    # Clear active alert id on connected FallDetector instances
    for det in active_detectors:
        if det.active_alert_id == alert_id:
            det.active_alert_id = None

    now_local = get_local_time_str()
    followup_msg = f"False alarm dismissed at {now_local}."
    target_chat = get_target_chat_id(client_id=cid)
    asyncio.create_task(send_telegram(followup_msg, chat_id=target_chat))

    return {
        **target_alert,
        "destination_chat_id": mask_chat_id(target_alert.get("destination_chat_id")),
        "chat_id_masked": mask_chat_id(target_alert.get("destination_chat_id")),
    }


@app.post("/test-alert")
async def test_alert(request: Request = None):
    """Creates a test alert, sends test Telegram message with rate limit, and returns result."""
    cid = get_client_id(request)
    body = {}
    if request is not None:
        try:
            body = await request.json()
        except Exception:
            body = {}

    if not isinstance(body, dict):
        body = {}

    is_test_clip = bool(body.get("is_test_clip"))

    if is_test_clip:
        # Check test clip rate limit: max 2 per hour
        if not check_test_clip_rate_limit(cid):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded: at most 2 test clips per hour allowed.",
            )
        # Ensure client has consented
        if not clip_consent.get(cid, False):
            raise HTTPException(
                status_code=403,
                detail="Turn on 'Send video clip' and tick the consent box, then try again.",
            )
    else:
        # Check test alert rate limit: max 3 per hour per client_id
        if not check_test_alert_rate_limit(cid):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded: at most 3 test alerts per hour allowed.",
            )

    custom_chat = None
    if body.get("telegram_chat_id"):
        validated = validate_chat_id(body.get("telegram_chat_id"))
        if not validated:
            raise HTTPException(
                status_code=400,
                detail="Invalid Telegram chat ID format. Must be numeric.",
            )
        custom_chat = validated

    target_chat = get_target_chat_id(client_id=cid, explicit_chat=custom_chat)
    alert_record = create_alert(
        probability=1.0,
        client_id=cid,
        is_test=True,
        destination_chat_id=target_chat,
        is_test_clip=is_test_clip,
        clip_status="pending" if is_test_clip else "disabled",
    )

    if not is_test_clip:
        test_msg = "FallGuard test alert. Your setup works."
        sent = await send_telegram(test_msg, chat_id=target_chat)
        alert_record["telegram_sent"] = sent
    else:
        sent = False
        alert_record["telegram_sent"] = False

    masked = mask_chat_id(target_chat)
    return {
        "status": "ok",
        "alert": {
            **alert_record,
            "destination_chat_id": masked,
            "chat_id_masked": masked,
        },
        "telegram_sent": sent,
        "chat_id_masked": masked,
        "masked_chat_id": masked,
        "message": (
            f"Test clip authorized (# {alert_record['id']})."
            if is_test_clip
            else (
                f"Test alert sent to Telegram chat {masked}."
                if sent
                else f"Telegram notification skipped/failed for chat {masked}."
            )
        ),
    }


@app.post("/api/telegram/verify")
async def verify_telegram_endpoint(request: Request):
    """Internal checks for Telegram Bot Token and Chat ID; sends verification message.

    Performs live checks against Telegram Bot API (getMe and sendMessage),
    ensures validity of credentials, and routes all future alerts to this chat ID.
    """
    cid = get_client_id(request)
    try:
        body = await request.json()
    except Exception:
        body = {}

    if not isinstance(body, dict):
        body = {}

    chat_id = str(body.get("chat_id", "")).strip()
    custom_token = str(body.get("token", "")).strip() if body.get("token") else None

    if not chat_id:
        raise HTTPException(status_code=400, detail="Telegram chat_id is required.")

    result = await asyncio.to_thread(_sync_verify_telegram, chat_id, custom_token)
    if not result.get("valid"):
        return JSONResponse(status_code=400, content=result)

    valid_chat = result["chat_id"]
    client_verified_chat_ids[cid] = valid_chat
    global active_verified_chat_id
    active_verified_chat_id = valid_chat

    return result


@app.post("/api/telegram/notify-error")
async def notify_error_endpoint(request: Request):
    """Notify caregiver via Telegram when an error occurs in monitoring or frontend."""
    cid = get_client_id(request)
    try:
        body = await request.json()
    except Exception:
        body = {}

    if not isinstance(body, dict):
        body = {}

    err_msg = str(body.get("error_message") or body.get("error") or "Unknown system error").strip()
    err_type = str(body.get("error_type") or "System").strip()
    chat_id = body.get("chat_id")

    sent = await send_system_error_telegram(
        error_message=err_msg,
        error_type=err_type,
        client_id=cid,
        chat_id=chat_id,
    )
    target = get_target_chat_id(client_id=cid, explicit_chat=chat_id)
    return {
        "status": "ok",
        "telegram_sent": sent,
        "chat_id_masked": mask_chat_id(target),
    }


@app.post("/api/consent/clip")
async def set_clip_consent(request: Request):
    """Set opt-in consent for incident video clip recording for the calling client."""
    cid = get_client_id(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="Request body must be valid JSON")

    if not isinstance(body, dict) or "consent" not in body or not isinstance(body["consent"], bool):
        raise HTTPException(
            status_code=422,
            detail="Field 'consent' must be an explicit boolean (true/false)",
        )

    val = body["consent"]
    clip_consent[cid] = val
    val_str = "true" if val else "false"
    logger.info(f"clip consent for {cid[:8]} = {val_str} (source: http)")
    return {"status": "ok", "client_id": cid, "consent": val, "clip_consent": val}


@app.post("/alerts/{alert_id}/clip")
async def upload_alert_clip(
    alert_id: int,
    request: Request,
    segments: List[UploadFile] = File(...),
    metadata: Optional[str] = Form(None),
    info: Optional[str] = Form(None),
):
    """Upload multipart 2-second video segments to process and dispatch 20s incident clip."""
    cid = get_client_id(request)

    # 1. Enforce client consent (HTTP 403)
    if not clip_consent.get(cid, False):
        raise HTTPException(
            status_code=403,
            detail="Turn on 'Send video clip' and tick the consent box, then try again.",
        )

    # 2. Check alert existence and ownership (HTTP 404)
    target_alert = None
    for a in in_memory_alerts:
        if a["id"] == alert_id:
            if a.get("client_id") == cid:
                target_alert = a
                break

    if target_alert is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found for this client")

    # 3. Check if alert is dismissed (HTTP 409)
    if target_alert.get("status") == "dismissed":
        target_alert["clip_status"] = "skipped_dismissed"
        raise HTTPException(status_code=409, detail="Alert has been dismissed; video clip discarded")

    # 4. Check if alert is an unauthorized test alert (HTTP 400)
    if target_alert.get("is_test") and not target_alert.get("is_test_clip"):
        raise HTTPException(status_code=400, detail="Incident clips cannot be uploaded for standard test alerts")

    # 5. Check if clip already sent/processing for this alert (HTTP 409)
    if target_alert.get("clip_status") in ("sent", "processing") or target_alert.get("clip_sent"):
        raise HTTPException(status_code=409, detail=f"Clip already processed for alert {alert_id}")

    # 6. Check hourly rate limit: at most 5 clips per hour per client_id (HTTP 429)
    if not check_clip_rate_limit(cid):
        target_alert["clip_status"] = "failed: rate limit exceeded"
        raise HTTPException(status_code=429, detail="Rate limit exceeded: maximum 5 clips per hour allowed")

    # 7. Check segment count: at most 14 segments, at least 1
    if not segments:
        raise HTTPException(status_code=400, detail="No video segments provided")
    if len(segments) > 14:
        raise HTTPException(status_code=413, detail=f"Too many segments ({len(segments)}): maximum 14 allowed")

    # 8. Check ffmpeg availability (HTTP 503)
    if not is_ffmpeg_available():
        target_alert["clip_status"] = "failed: ffmpeg not available"
        raise HTTPException(status_code=503, detail="Video processing unavailable: ffmpeg is not installed on server")

    # 9. Parse metadata JSON
    meta_raw = metadata or info or "{}"
    try:
        meta_data = json.loads(meta_raw)
        if not isinstance(meta_data, dict):
            meta_data = {}
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid clip metadata JSON")

    T_ms = meta_data.get("T")
    if T_ms is None or not isinstance(T_ms, (int, float)):
        T_ms = time.time() * 1000.0

    seg_meta_list = meta_data.get("segments", [])

    # Mark as processing
    target_alert["clip_status"] = "processing"

    # Process in temporary directory and ensure cleanup in finally block
    with tempfile.TemporaryDirectory() as tmp_dir:
        segment_files = []
        segment_start_times = []
        total_upload_bytes = 0

        for idx, seg in enumerate(segments):
            # Validate media type: only video/webm or video/mp4 (HTTP 415)
            raw_ctype = (seg.content_type or "").lower().split(";")[0].strip()
            if raw_ctype not in ("video/webm", "video/mp4", "video/x-matroska"):
                target_alert["clip_status"] = f"failed: unsupported media type {raw_ctype}"
                raise HTTPException(
                    status_code=415,
                    detail=f"Unsupported media type: {seg.content_type}. Only video/webm or video/mp4 allowed",
                )

            # Read content and enforce segment size limit: at most 3 MB (HTTP 413)
            content = await seg.read()
            seg_len = len(content)
            if seg_len > 3 * 1024 * 1024:
                target_alert["clip_status"] = "failed: segment exceeded 3MB"
                raise HTTPException(
                    status_code=413,
                    detail=f"Segment {idx} ({seg_len / 1024 / 1024:.2f} MB) exceeds maximum 3 MB limit",
                )

            total_upload_bytes += seg_len
            if total_upload_bytes > 15 * 1024 * 1024:
                target_alert["clip_status"] = "failed: total upload exceeded 15MB"
                raise HTTPException(
                    status_code=413,
                    detail="Total clip payload exceeds maximum 15 MB limit",
                )

            ext = ".webm" if "webm" in raw_ctype or "matroska" in raw_ctype else ".mp4"
            seg_filename = os.path.join(tmp_dir, f"segment_{idx:03d}{ext}")
            with open(seg_filename, "wb") as f:
                f.write(content)
            segment_files.append(seg_filename)

            # Extract start time from metadata or fallback
            seg_start_ms = None
            if idx < len(seg_meta_list) and isinstance(seg_meta_list[idx], dict):
                seg_start_ms = seg_meta_list[idx].get("start_ms")
            if seg_start_ms is None or not isinstance(seg_start_ms, (int, float)):
                seg_start_ms = float(T_ms) - (len(segments) - idx) * 2000.0
            segment_start_times.append(float(seg_start_ms))

        # Check alert dismissal status again before running ffmpeg
        if target_alert.get("status") == "dismissed":
            target_alert["clip_status"] = "skipped_dismissed"
            raise HTTPException(status_code=409, detail="Alert dismissed during upload; skipped clip processing")

        output_clip_path = os.path.join(tmp_dir, "incident_clip.mp4")

        # Concurrency limit: 1 worker
        async with clip_processing_semaphore:
            try:
                await process_incident_clip(
                    segment_paths=segment_files,
                    segment_start_times=segment_start_times,
                    T_ms=float(T_ms),
                    output_path=output_clip_path,
                )
            except TimeoutError as te:
                target_alert["clip_status"] = "failed: ffmpeg timeout (40s)"
                raise HTTPException(status_code=504, detail=str(te))
            except ValueError as ve:
                target_alert["clip_status"] = f"failed: {ve}"
                raise HTTPException(status_code=413, detail=str(ve))
            except Exception as exc:
                target_alert["clip_status"] = f"failed: {exc}"
                raise HTTPException(status_code=500, detail=f"Failed to process video clip: {exc}")

            # Re-check dismissal before Telegram delivery
            if target_alert.get("status") == "dismissed":
                target_alert["clip_status"] = "skipped_dismissed"
                raise HTTPException(status_code=409, detail="Alert dismissed; skipping Telegram video delivery")

            # Resolve destination chat_id: exact chat used for alert text
            dest_chat = target_alert.get("destination_chat_id") or get_target_chat_id(client_id=cid)
            if not dest_chat:
                target_alert["clip_status"] = "failed: no chat_id"
                raise HTTPException(status_code=400, detail="No destination Telegram chat_id registered for alert")

            # Send video to caregiver's Telegram chat
            now_local = target_alert.get("local_time") or get_local_time_str()
            sent = await send_telegram_video(output_clip_path, chat_id=dest_chat, local_time=now_local)

            if sent:
                target_alert["clip_status"] = "sent"
                target_alert["clip_sent"] = True
                logger.info(f"Incident clip sent successfully for alert {alert_id} to {mask_chat_id(dest_chat)}.")
                return {
                    "status": "ok",
                    "alert_id": alert_id,
                    "clip_status": "sent",
                    "destination_chat_masked": mask_chat_id(dest_chat),
                    "message": f"Incident clip delivered to Telegram chat {mask_chat_id(dest_chat)}.",
                }
            else:
                target_alert["clip_status"] = "failed: telegram delivery failed"
                return JSONResponse(
                    status_code=502,
                    content={
                        "status": "error",
                        "alert_id": alert_id,
                        "clip_status": "failed: telegram delivery failed",
                        "detail": "Failed to deliver video clip to Telegram Bot API.",
                    },
                )


# ---------------------------------------------------------------------------
# WebSocket Validation & Ingestion Endpoint
# ---------------------------------------------------------------------------
def validate_landmarks_payload(data: Any) -> bool:
    """Validate message structure: {"t": int/float, "landmarks": [[x,y,z,visibility] x 33]}.

    Ignores bad messages gracefully without raising unhandled exceptions.
    """
    if not isinstance(data, dict):
        return False

    landmarks = data.get("landmarks")
    if not isinstance(landmarks, list) or len(landmarks) != 33:
        return False

    for pt in landmarks:
        if not isinstance(pt, (list, tuple)) or len(pt) != 4:
            return False
        for val in pt:
            if not isinstance(val, (int, float)):
                return False

    return True


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint ingesting 33-landmark pose telemetry.

    Validates payload format, evaluates causal FallDetector pipeline, and returns status.
    """
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    client_port = websocket.client.port if websocket.client else "unknown"
    logger.info(f"WebSocket client connected from {client_host}:{client_port}")

    client_id = websocket.query_params.get("client_id", "anonymous").strip() or "anonymous"
    connection_chat_id: Optional[str] = client_verified_chat_ids.get(client_id) or active_verified_chat_id

    # Create dedicated FallDetector instance for this connection
    detector = FallDetector(
        model_name=MODEL_NAME,
        probability_threshold=PROBABILITY_THRESHOLD,
        lying_seconds=LYING_SECONDS,
    )
    active_detectors.append(detector)
    last_client_t_ms: float = 0.0

    try:
        while True:
            try:
                # Receive client message with 1.0s timeout to detect missing person gaps
                text_data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                # 1.0s elapsed with no frames received -> trigger missing person timeout check
                timeout_res = detector.check_timeout()
                if timeout_res["status"] == "alert":
                    alert_record = create_alert(probability=timeout_res["probability"], client_id=client_id)
                    timeout_res["alert_id"] = alert_record["id"]
                    detector.active_alert_id = alert_record["id"]
                    asyncio.create_task(dispatch_fall_telegram(alert_record, chat_id=connection_chat_id))
                    logger.warning(f"ALERT registered during timeout for client {client_id}: {alert_record}")
                    await websocket.send_json(timeout_res)
                continue

            try:
                data = json.loads(text_data)
            except json.JSONDecodeError:
                logger.warning("Received malformed JSON on /ws; ignoring message.")
                continue

            # Handle per-connection settings update
            if isinstance(data, dict) and data.get("type") == "settings":
                detector.apply_settings(data)
                raw_chat = data.get("telegram_chat_id")
                if raw_chat is not None and str(raw_chat).strip():
                    valid_chat = validate_chat_id(raw_chat)
                    if valid_chat:
                        connection_chat_id = valid_chat
                        client_verified_chat_ids[client_id] = valid_chat
                        logger.info(f"Updated per-connection chat ID for {client_id}: {mask_chat_id(valid_chat)}")
                    else:
                        logger.warning(f"Rejected invalid Telegram chat ID from {client_id}: {raw_chat}")
                elif raw_chat == "":
                    connection_chat_id = None
                    if client_id in client_verified_chat_ids:
                        del client_verified_chat_ids[client_id]

                # Update clip consent if present in settings using explicit boolean field
                raw_ws_consent = None
                if "consent" in data:
                    raw_ws_consent = data["consent"]
                elif "clip_consent" in data:
                    raw_ws_consent = data["clip_consent"]

                if isinstance(raw_ws_consent, bool):
                    clip_consent[client_id] = raw_ws_consent
                    val_str = "true" if raw_ws_consent else "false"
                    logger.info(f"clip consent for {client_id[:8]} = {val_str} (source: ws)")

                await websocket.send_json({
                    "type": "settings_ack",
                    "threshold": detector.threshold,
                    "lying_seconds": detector.lying_seconds,
                    "clip_consent": clip_consent.get(client_id, False),
                    "consent": clip_consent.get(client_id, False),
                })
                continue

            # Validate 33 landmarks with [x, y, z, visibility]
            if not validate_landmarks_payload(data):
                logger.debug("Received invalid landmark structure (expected 33x4); ignoring.")
                continue

            # Timestamp validation: accept client timestamp, fallback to server time if missing or backwards
            now_ms = time.time() * 1000.0
            raw_t = data.get("t")
            if raw_t is None or not isinstance(raw_t, (int, float)) or float(raw_t) <= last_client_t_ms:
                effective_t_ms = now_ms
            else:
                effective_t_ms = float(raw_t)
            last_client_t_ms = effective_t_ms

            # Feed into FallDetector
            result = detector.add_frame(effective_t_ms, data["landmarks"])

            # If alert confirmed, register in in-memory alerts and send notification
            if result["status"] == "alert":
                alert_record = create_alert(probability=result["probability"], client_id=client_id)
                result["alert_id"] = alert_record["id"]
                detector.active_alert_id = alert_record["id"]
                asyncio.create_task(dispatch_fall_telegram(alert_record, chat_id=connection_chat_id))
                logger.warning(f"ALERT registered for client {client_id}: {alert_record}")

            # Send response to client
            await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {client_host}:{client_port}")
    except Exception as exc:
        logger.error(f"Unexpected WebSocket error with {client_host}:{client_port}: {exc}")
        asyncio.create_task(
            send_system_error_telegram(
                f"WebSocket server error: {exc}",
                error_type="WebSocket",
                client_id=client_id,
                chat_id=connection_chat_id,
            )
        )
    finally:
        if detector in active_detectors:
            active_detectors.remove(detector)


# ---------------------------------------------------------------------------
# Static Frontend Files Mount (Serves website at root "/")
# ---------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    logger.info(f"Serving static frontend files from: {FRONTEND_DIR}")
else:
    logger.warning(f"Frontend directory not found at: {FRONTEND_DIR}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.backend.main:app", host="0.0.0.0", port=PORT, reload=True)
