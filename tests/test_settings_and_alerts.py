"""
Tests for FallGuard Settings, Telegram Safety, Client Isolation, and Rate Limiting.
Meets Requirement 6:
- Settings validation (out-of-range and malformed values)
- Chat ID validation & fallback
- Client isolation of /alerts
- Rate limiting (with Telegram mocked, no real network calls)
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch
import pytest
from fastapi import HTTPException

from app.backend.detector import FallDetector
from app.backend.main import (
    validate_chat_id,
    mask_chat_id,
    check_real_alert_rate_limit,
    check_test_alert_rate_limit,
    create_alert,
    in_memory_alerts,
    client_real_alert_timestamps,
    client_test_alert_timestamps,
    get_alerts,
    clear_alerts,
    test_alert as api_test_alert,
)


class MockRequest:
    """Lightweight Starlette-compatible request mock avoiding deprecated TestClient warning."""

    def __init__(self, headers=None, json_data=None):
        self.headers = headers or {}
        self._json_data = json_data

    async def json(self):
        return self._json_data or {}


@pytest.fixture(autouse=True)
def reset_backend_state():
    """Reset alerts, rate limit caches, and active detectors before each test."""
    in_memory_alerts.clear()
    client_real_alert_timestamps.clear()
    client_test_alert_timestamps.clear()
    yield
    in_memory_alerts.clear()
    client_real_alert_timestamps.clear()
    client_test_alert_timestamps.clear()


class TestSettingsValidation:
    """Verifies settings clamping, fallback, and malformed value tolerance."""

    def test_detector_settings_clamping(self):
        detector = FallDetector()
        saved = detector.saved_threshold

        # Default sensitivity 3
        detector.apply_settings({"sensitivity": 3, "lying_seconds": 2.0})
        assert detector.threshold == pytest.approx(saved, abs=1e-3)
        assert detector.lying_seconds == 2.0

        # Sensitivity 1 (lower sensitivity -> higher threshold)
        detector.apply_settings({"sensitivity": 1})
        expected_thr1 = max(0.30, min(0.95, saved + 0.20))
        assert detector.threshold == pytest.approx(expected_thr1, abs=1e-3)

        # Sensitivity 5 (higher sensitivity -> lower threshold)
        detector.apply_settings({"sensitivity": 5})
        expected_thr5 = max(0.30, min(0.95, saved - 0.20))
        assert detector.threshold == pytest.approx(expected_thr5, abs=1e-3)

        # Out-of-range sensitivity clamped to [1, 5]
        detector.apply_settings({"sensitivity": -10})
        assert detector.threshold == pytest.approx(expected_thr1, abs=1e-3)

        detector.apply_settings({"sensitivity": 99})
        assert detector.threshold == pytest.approx(expected_thr5, abs=1e-3)

        # Out-of-range lying_seconds clamped to [1.0, 5.0]
        detector.apply_settings({"lying_seconds": -5.0})
        assert detector.lying_seconds == 1.0

        detector.apply_settings({"lying_seconds": 42.0})
        assert detector.lying_seconds == 5.0

    def test_detector_malformed_settings_ignored(self):
        detector = FallDetector()
        orig_thr = detector.threshold
        orig_lying = detector.lying_seconds

        # Malformed values must never crash the detector
        detector.apply_settings({"sensitivity": "not-a-number", "lying_seconds": None})
        assert detector.threshold == orig_thr
        assert detector.lying_seconds == orig_lying

        detector.apply_settings("completely invalid non-dict payload")  # type: ignore
        assert detector.threshold == orig_thr
        assert detector.lying_seconds == orig_lying


class TestChatIdValidation:
    """Verifies Telegram chat ID validation, masking, and fallback."""

    def test_valid_chat_ids(self):
        assert validate_chat_id("5513741585") == "5513741585"
        assert validate_chat_id("-100123456789") == "-100123456789"
        assert validate_chat_id("1") == "1"
        assert validate_chat_id(-100987654321) == "-100987654321"

    def test_invalid_chat_ids(self):
        assert validate_chat_id("abc") is None
        assert validate_chat_id("551374a1585") is None
        assert validate_chat_id("123; DROP TABLE") is None
        assert validate_chat_id("") is None
        assert validate_chat_id("   ") is None
        assert validate_chat_id(None) is None
        assert validate_chat_id([]) is None

    def test_chat_id_masking(self):
        assert mask_chat_id("5513741585") == "...1585"
        assert mask_chat_id("-100123456789") == "...6789"
        assert mask_chat_id("12") == "...12"
        assert mask_chat_id("") == ""
        assert mask_chat_id(None) == ""


class TestClientIsolationOfAlerts:
    """Verifies that visitors never see each other's alerts and clear only their own."""

    def test_get_alerts_client_isolation(self):
        # Create alerts for client-A and client-B
        create_alert(probability=0.88, client_id="client-A", status="confirmed")
        create_alert(probability=0.92, client_id="client-B", status="confirmed")
        create_alert(probability=0.85, client_id="client-A", status="confirmed")

        # GET /alerts for client-A
        req_a = MockRequest(headers={"X-Client-Id": "client-A"})
        alerts_a = asyncio.run(get_alerts(req_a))
        assert len(alerts_a) == 2
        for a in alerts_a:
            assert a["client_id"] == "client-A"
        # Verify newest first ordering
        assert alerts_a[0]["peak_probability"] == 0.85
        assert alerts_a[1]["peak_probability"] == 0.88

        # GET /alerts for client-B
        req_b = MockRequest(headers={"X-Client-Id": "client-B"})
        alerts_b = asyncio.run(get_alerts(req_b))
        assert len(alerts_b) == 1
        assert alerts_b[0]["client_id"] == "client-B"
        assert alerts_b[0]["peak_probability"] == 0.92

        # Client-C with no alerts
        req_c = MockRequest(headers={"X-Client-Id": "client-C"})
        alerts_c = asyncio.run(get_alerts(req_c))
        assert alerts_c == []

    def test_clear_alerts_client_isolation(self):
        create_alert(probability=0.80, client_id="client-A")
        create_alert(probability=0.85, client_id="client-B")
        create_alert(probability=0.90, client_id="client-A")

        # Clear alerts for client-A
        req_clear = MockRequest(headers={"X-Client-Id": "client-A"})
        res_clear = asyncio.run(clear_alerts(req_clear))
        assert res_clear["cleared_count"] == 2

        # Client-A should now have 0 alerts
        req_a = MockRequest(headers={"X-Client-Id": "client-A"})
        assert asyncio.run(get_alerts(req_a)) == []

        # Client-B must still retain their alert intact
        req_b = MockRequest(headers={"X-Client-Id": "client-B"})
        alerts_b = asyncio.run(get_alerts(req_b))
        assert len(alerts_b) == 1
        assert alerts_b[0]["client_id"] == "client-B"
        assert alerts_b[0]["peak_probability"] == 0.85


class TestRateLimitingWithMockedTelegram:
    """Verifies 1 real alert / 60s and 3 test alerts / hour rate limits."""

    def test_real_alert_rate_limiting(self):
        cid = "client-safety-1"
        assert check_real_alert_rate_limit(cid) is True

        # Second attempt immediately should be rejected
        assert check_real_alert_rate_limit(cid) is False

        # Another client should not be blocked
        assert check_real_alert_rate_limit("client-safety-2") is True

        # After 61 seconds, client-safety-1 should be allowed again
        client_real_alert_timestamps[cid] = time.time() - 61.0
        assert check_real_alert_rate_limit(cid) is True

    @patch("app.backend.main.send_telegram", new_callable=AsyncMock)
    def test_test_alert_rate_limiting_and_error(self, mock_send):
        mock_send.return_value = True
        cid = "client-test-limit"
        req = MockRequest(headers={"X-Client-Id": cid})

        # 1st test alert -> OK
        r1 = asyncio.run(api_test_alert(req))
        assert r1["status"] == "ok"
        assert r1["telegram_sent"] is True

        # 2nd test alert -> OK
        r2 = asyncio.run(api_test_alert(req))
        assert r2["status"] == "ok"

        # 3rd test alert -> OK
        r3 = asyncio.run(api_test_alert(req))
        assert r3["status"] == "ok"

        # 4th test alert within the hour -> 429 Too Many Requests
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(api_test_alert(req))
        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in exc_info.value.detail

        # A different client_id can still send their test alerts
        req_other = MockRequest(headers={"X-Client-Id": "other-client"})
        r_other = asyncio.run(api_test_alert(req_other))
        assert r_other["status"] == "ok"

    @patch("app.backend.main.send_telegram", new_callable=AsyncMock)
    def test_custom_chat_id_in_test_alert_shows_masked(self, mock_send):
        mock_send.return_value = True
        cid = "client-custom-chat"

        # Valid custom chat ID
        req_valid = MockRequest(
            headers={"X-Client-Id": cid},
            json_data={"telegram_chat_id": "5513741585"},
        )
        resp = asyncio.run(api_test_alert(req_valid))
        assert resp["status"] == "ok"
        assert resp["masked_chat_id"] == "...1585"

        # Invalid custom chat ID must be rejected with 400
        req_bad = MockRequest(
            headers={"X-Client-Id": cid},
            json_data={"telegram_chat_id": "malformed_id"},
        )
        with pytest.raises(HTTPException) as exc_bad:
            asyncio.run(api_test_alert(req_bad))
        assert exc_bad.value.status_code == 400
        assert "Invalid Telegram chat ID" in exc_bad.value.detail
