"""test_telegram_verification.py - Test suite for Telegram Bot Token & Chat ID verification and Error notification.

Covers:
1. Internal check: Missing Telegram bot token returns valid=False.
2. Internal check: Invalid Telegram bot token (401) returns valid=False without blindly trusting.
3. Internal check: Malformed non-numeric chat ID returns valid=False.
4. Internal check: Unreachable chat ID (400 chat not found) returns valid=False with helpful hint.
5. Internal check: Successful verification sends test message, returns valid=True, saves verified chat ID.
6. Fall alerts, test alerts, and dismissals route to verified chat ID only.
7. Error notification endpoint (POST /api/telegram/notify-error) sends formatted error alert to verified chat ID with rate limiting.
"""

import asyncio
from unittest.mock import MagicMock, patch
import pytest
from fastapi import HTTPException

from app.backend.main import (
    _sync_verify_telegram,
    client_verified_chat_ids,
    active_verified_chat_id,
    client_error_alert_timestamps,
    get_target_chat_id,
    send_system_error_telegram,
    check_error_alert_rate_limit,
    in_memory_alerts,
    create_alert,
    dispatch_fall_telegram,
    test_alert as api_test_alert,
    verify_telegram_endpoint,
    notify_error_endpoint,
)


class MockRequest:
    """Mock Starlette/FastAPI request for testing endpoints."""

    def __init__(self, headers=None, json_data=None):
        self.headers = headers or {}
        self._json_data = json_data

    async def json(self):
        return self._json_data or {}


@pytest.fixture(autouse=True)
def clean_state():
    """Reset alerts, timestamps, and verified chat IDs before each test."""
    client_verified_chat_ids.clear()
    client_error_alert_timestamps.clear()
    in_memory_alerts.clear()
    yield
    client_verified_chat_ids.clear()
    client_error_alert_timestamps.clear()
    in_memory_alerts.clear()


class TestTelegramVerification:
    """Tests for Telegram Bot Token and Chat ID internal check verification."""

    def test_missing_token_returns_invalid(self):
        """When bot token is empty, verify fails immediately without saying valid."""
        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": ""}):
            result = _sync_verify_telegram(chat_id="5513741585", custom_token="")
            assert result["valid"] is False
            assert "missing" in result["error"].lower()

    @patch("requests.get")
    def test_invalid_token_returns_invalid(self, mock_get):
        """When Telegram Bot API rejects token with 401 Unauthorized, verify fails."""
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"ok": False, "error_code": 401, "description": "Unauthorized"}
        mock_get.return_value = mock_resp

        result = _sync_verify_telegram(chat_id="5513741585", custom_token="invalid_bot_token")
        assert result["valid"] is False
        assert "401" in result["error"] or "unauthorized" in result["error"].lower()

    @patch("requests.get")
    def test_invalid_chat_id_format(self, mock_get):
        """Non-numeric chat ID fails format validation before sending message."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "ok": True,
            "result": {"id": 123456, "is_bot": True, "first_name": "FallGuard", "username": "FallGuardBot"},
        }
        mock_get.return_value = mock_resp

        result = _sync_verify_telegram(chat_id="not_a_number", custom_token="valid_token")
        assert result["valid"] is False
        assert "numeric" in result["error"].lower()

    @patch("requests.post")
    @patch("requests.get")
    def test_chat_not_found_returns_invalid_with_hint(self, mock_get, mock_post):
        """When user hasn't started the bot, Telegram returns 400 chat not found."""
        mock_get_resp = MagicMock()
        mock_get_resp.status_code = 200
        mock_get_resp.json.return_value = {
            "ok": True,
            "result": {"id": 123456, "is_bot": True, "first_name": "FallGuard", "username": "FallGuardBot"},
        }
        mock_get.return_value = mock_get_resp

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 400
        mock_post_resp.json.return_value = {
            "ok": False,
            "error_code": 400,
            "description": "Bad Request: chat not found",
        }
        mock_post.return_value = mock_post_resp

        result = _sync_verify_telegram(chat_id="999999999", custom_token="valid_token")
        assert result["valid"] is False
        assert "chat not found" in result["error"].lower()
        assert "start" in result["error"].lower()

    @patch("requests.post")
    @patch("requests.get")
    def test_successful_verification(self, mock_get, mock_post):
        """When getMe and sendMessage succeed, verification passes and returns details."""
        mock_get_resp = MagicMock()
        mock_get_resp.status_code = 200
        mock_get_resp.json.return_value = {
            "ok": True,
            "result": {"id": 123456, "is_bot": True, "first_name": "FallGuard Alerts", "username": "FallGuard_bot"},
        }
        mock_get.return_value = mock_get_resp

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {"ok": True, "result": {"message_id": 42}}
        mock_post.return_value = mock_post_resp

        result = _sync_verify_telegram(chat_id="5513741585", custom_token="valid_token")
        assert result["valid"] is True
        assert result["bot_username"] == "FallGuard_bot"
        assert result["chat_id"] == "5513741585"
        assert result["masked_chat_id"] == "...1585"
        assert "valid" in result["message"].lower()

    @pytest.mark.anyio
    @patch("app.backend.main._sync_verify_telegram")
    async def test_verify_endpoint_saves_verified_chat(self, mock_sync_verify):
        """POST /api/telegram/verify saves verified chat ID for the client and session."""
        mock_sync_verify.return_value = {
            "ok": True,
            "valid": True,
            "chat_id": "77889900",
            "masked_chat_id": "...8900",
            "bot_username": "FallGuardBot",
            "bot_name": "FallGuard",
            "message": "Token is valid and verification message sent.",
        }

        req = MockRequest(headers={"X-Client-Id": "client-abc"}, json_data={"chat_id": "77889900"})
        res = await verify_telegram_endpoint(req)

        assert res["valid"] is True
        assert client_verified_chat_ids.get("client-abc") == "77889900"
        assert get_target_chat_id(client_id="client-abc") == "77889900"


class TestTelegramMessageRouting:
    """Verifies that alerts and errors route to verified chat ID only."""

    @pytest.mark.anyio
    @patch("app.backend.main.send_telegram")
    async def test_fall_alert_routes_to_verified_chat(self, mock_send):
        """When client has verified chat ID, fall alert sends message to this chat ID only."""
        mock_send.return_value = True
        client_verified_chat_ids["client-xyz"] = "987654321"

        record = create_alert(probability=0.95, client_id="client-xyz")
        await dispatch_fall_telegram(record)

        mock_send.assert_called_once()
        _, kwargs = mock_send.call_args
        assert kwargs["chat_id"] == "987654321"

    @pytest.mark.anyio
    @patch("app.backend.main.send_telegram")
    async def test_notify_error_endpoint(self, mock_send):
        """POST /api/telegram/notify-error sends error alert to verified chat ID."""
        mock_send.return_value = True
        client_verified_chat_ids["client-err"] = "11223344"

        req = MockRequest(
            headers={"X-Client-Id": "client-err"},
            json_data={"error_message": "Camera access denied by user", "error_type": "Camera"},
        )
        res = await notify_error_endpoint(req)

        assert res["status"] == "ok"
        assert res["telegram_sent"] is True
        mock_send.assert_called_once()
        call_msg = mock_send.call_args[0][0]
        assert "Camera access denied by user" in call_msg
        assert "CAMERA" in call_msg.upper()
        assert mock_send.call_args[1]["chat_id"] == "11223344"

    @pytest.mark.anyio
    @patch("app.backend.main.send_telegram")
    async def test_error_alert_rate_limiting(self, mock_send):
        """Error notifications are rate limited (max 1 per 30s) to prevent spamming."""
        mock_send.return_value = True
        client_verified_chat_ids["client-spam"] = "55556666"

        sent1 = await send_system_error_telegram("Error 1", client_id="client-spam")
        assert sent1 is True

        # Second call immediately after must be suppressed by rate limit
        sent2 = await send_system_error_telegram("Error 2", client_id="client-spam")
        assert sent2 is False
        assert mock_send.call_count == 1
