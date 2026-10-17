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

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.backend.detector import FallDetector, load_model_bundle

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


def create_alert(
    probability: float = 1.0,
    client_id: str = "anonymous",
    is_test: bool = False,
    status: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Helper to register a new alert in in-memory storage."""
    prob = kwargs.get("peak_prob", kwargs.get("prob", probability))
    now_utc = datetime.now(timezone.utc)
    alert_id = len(in_memory_alerts) + 1
    local_time = get_local_time_str(now_utc)
    stat = status or ("test" if is_test else "confirmed")
    record: Dict[str, Any] = {
        "id": alert_id,
        "client_id": client_id,
        "time": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "local_time": local_time,
        "probability": round(float(prob), 4),
        "peak_probability": round(float(prob), 4),
        "status": stat,
        "telegram_sent": False,
    }
    if is_test:
        record["is_test"] = True
    in_memory_alerts.append(record)
    return record


async def dispatch_fall_telegram(record: Dict[str, Any], chat_id: Optional[str] = None) -> None:
    """Send Telegram message for confirmed fall alert and record delivery status."""
    client_id = record.get("client_id", "anonymous")
    if not check_real_alert_rate_limit(client_id):
        logger.warning(f"Rate limit: real alert Telegram suppressed for client {client_id} (max 1 per 60s).")
        record["telegram_sent"] = False
        return

    msg = f"FALL DETECTED at {record['local_time']}. Please check on them."
    sent = await send_telegram(msg, chat_id=chat_id)
    record["telegram_sent"] = sent


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
        "disclaimer": "Prototype for educational/research purposes only. Not a certified medical device.",
    }


@app.get("/alerts")
async def get_alerts(request: Request = None):
    """Returns in-memory list of verified fall alerts for the calling client_id (newest first)."""
    if request is None:
        return list(reversed(in_memory_alerts))
    cid = get_client_id(request)
    client_alerts = [a for a in reversed(in_memory_alerts) if a.get("client_id") == cid]
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

    # Clear active alert id on connected FallDetector instances
    for det in active_detectors:
        if det.active_alert_id == alert_id:
            det.active_alert_id = None

    now_local = get_local_time_str()
    followup_msg = f"False alarm dismissed at {now_local}."
    asyncio.create_task(send_telegram(followup_msg))

    return target_alert


@app.post("/test-alert")
async def test_alert(request: Request = None):
    """Creates a test alert, sends test Telegram message with rate limit, and returns result."""
    cid = get_client_id(request)

    # Check test alert rate limit: max 3 per hour per client_id
    if not check_test_alert_rate_limit(cid):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded: at most 3 test alerts per hour allowed.",
        )

    custom_chat = None
    if request is not None:
        try:
            body = await request.json()
            if isinstance(body, dict) and body.get("telegram_chat_id"):
                validated = validate_chat_id(body.get("telegram_chat_id"))
                if not validated:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid Telegram chat ID format. Must be numeric.",
                    )
                custom_chat = validated
        except HTTPException:
            raise
        except Exception:
            pass

    alert_record = create_alert(probability=1.0, client_id=cid, is_test=True)
    test_msg = "FallGuard test alert. Your setup works."

    target_chat = custom_chat or os.getenv("TELEGRAM_CHAT_ID", "").strip()
    sent = await send_telegram(test_msg, chat_id=target_chat)
    alert_record["telegram_sent"] = sent

    masked = mask_chat_id(target_chat)
    return {
        "status": "ok",
        "alert": alert_record,
        "telegram_sent": sent,
        "chat_id_masked": masked,
        "masked_chat_id": masked,
        "message": (
            f"Test alert sent to Telegram chat {masked}."
            if sent
            else f"Telegram notification skipped/failed for chat {masked}."
        ),
    }


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
    connection_chat_id: Optional[str] = None

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
                        logger.info(f"Updated per-connection chat ID for {client_id}: {mask_chat_id(valid_chat)}")
                    else:
                        logger.warning(f"Rejected invalid Telegram chat ID from {client_id}: {raw_chat}")
                elif raw_chat == "":
                    connection_chat_id = None

                await websocket.send_json({
                    "type": "settings_ack",
                    "threshold": detector.threshold,
                    "lying_seconds": detector.lying_seconds,
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
