"""test_alerting.py - Test Suite for Alerting, Telegram Integration, and State Machine Fixes.

Covers:
1. Peak-probability fix in FallDetector:
   - Stores the peak probability seen since possible fall began (not confirmation probability).
   - Validates alert response includes peak probability and alert_id when active.
2. Alert dismissal endpoint (POST /alerts/{id}/dismiss):
   - Successfully dismisses an existing alert and clears active_alert_id.
   - Returns 404 for unknown alert IDs.
3. POST /test-alert endpoint with mocked Telegram:
   - Ensures no real network requests are made.
   - Verifies test alert creation, newest-first ordering on GET /alerts.
4. Missing or invalid Telegram token safety:
   - Verifies missing credentials return False without crashing or raising exceptions.
   - Verifies exceptions sanitize token from logs and error strings.
"""

import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from fastapi import HTTPException

from app.backend.detector import FallDetector
from app.backend.main import (
    active_detectors,
    create_alert,
    dismiss_alert,
    get_alerts,
    get_local_time_str,
    in_memory_alerts,
    send_telegram,
    test_alert as api_test_alert,
)


class TestAlerting(unittest.TestCase):
    """Test suite for FallGuard alerting system, state machine peak probability, and endpoints."""

    def setUp(self):
        # Clear in-memory alert registry and active detectors before each test
        in_memory_alerts.clear()
        active_detectors.clear()

    def test_01_peak_probability_fix_during_possible_fall(self):
        """Verify that FallDetector records PEAK probability, not confirmation-time probability."""
        detector = FallDetector()
        detector.threshold = 0.60
        detector.lying_seconds = 1.0

        t = 100.0
        # 1. Start in normal state
        detector.status = "normal"
        detector.current_probability = 0.10
        detector.last_state_eval_t = t

        # 2. Trigger possible_fall with high peak probability (0.95)
        t += 0.25
        detector.current_probability = 0.95
        detector._update_state_machine(t_curr=t, new_inference=True)

        self.assertEqual(detector.status, "possible_fall")
        self.assertAlmostEqual(detector.peak_probability, 0.95, places=3)

        # 3. Subsequent frames during lying have lower probability (0.35)
        t += 0.5
        detector.current_probability = 0.35
        detector.last_torso_angle_mean = 75.0  # Lying angle (> 60)
        detector.last_hip_height_mean = 0.20   # Lying height (< 0.5)
        detector._update_state_machine(t_curr=t, new_inference=True)

        self.assertEqual(detector.status, "possible_fall")
        # Peak probability MUST still be 0.95, not lowered to 0.35
        self.assertAlmostEqual(detector.peak_probability, 0.95, places=3)

        # 4. Exceed lying confirmation duration to reach ALERT
        t += 0.6
        detector.current_probability = 0.25
        detector._update_state_machine(t_curr=t, new_inference=True)

        self.assertEqual(detector.status, "alert")
        resp = detector._build_response()

        # The alert response MUST report the peak probability (0.95), not current (0.25)
        self.assertAlmostEqual(resp["probability"], 0.95, places=2)
        self.assertAlmostEqual(resp.get("peak_probability", 0.0), 0.95, places=2)

    def test_02_alert_id_included_when_active(self):
        """Verify that 'alert_id' is included in WebSocket response whenever an alert is active."""
        detector = FallDetector()
        detector.active_alert_id = 42
        detector.status = "normal"

        resp = detector._build_response()
        self.assertEqual(resp.get("alert_id"), 42)

        # Clear alert ID on dismissal
        detector.active_alert_id = None
        resp_cleared = detector._build_response()
        self.assertNotIn("alert_id", resp_cleared)

    def test_03_dismiss_alert_success(self):
        """Test POST /alerts/{id}/dismiss updates alert status and notifies caregiver."""
        alert = create_alert(probability=0.88)
        self.assertEqual(alert["status"], "confirmed")
        alert_id = alert["id"]

        detector = FallDetector()
        detector.active_alert_id = alert_id
        active_detectors.append(detector)

        with patch("app.backend.main.send_telegram", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = True

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                res = loop.run_until_complete(dismiss_alert(alert_id))
            finally:
                loop.close()

        self.assertEqual(res["status"], "dismissed")
        self.assertEqual(alert["status"], "dismissed")
        # Active detector's active_alert_id should be cleared
        self.assertIsNone(detector.active_alert_id)

    def test_04_dismiss_unknown_alert_id_returns_404(self):
        """Test POST /alerts/{id}/dismiss with non-existent ID raises HTTP 404."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with self.assertRaises(HTTPException) as ctx:
                loop.run_until_complete(dismiss_alert(999999))
            self.assertEqual(ctx.exception.status_code, 404)
        finally:
            loop.close()

    def test_05_test_alert_endpoint_with_mocked_telegram(self):
        """Test POST /test-alert creates an alert and mocks Telegram without network calls."""
        with patch("app.backend.main.send_telegram", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = True

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(api_test_alert())
                alerts_newest = loop.run_until_complete(get_alerts())
            finally:
                loop.close()

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["telegram_sent"])
        self.assertTrue(result["alert"].get("is_test"))

        # Verify newest-first ordering on GET /alerts
        self.assertGreaterEqual(len(alerts_newest), 1)
        self.assertEqual(alerts_newest[0]["id"], result["alert"]["id"])

    def test_06_missing_telegram_token_does_not_crash(self):
        """Test that missing TELEGRAM_BOT_TOKEN or CHAT_ID returns False gracefully."""
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Direct call to send_telegram
                sent = loop.run_until_complete(send_telegram("FallGuard test message"))
                self.assertFalse(sent)

                # Call test_alert endpoint with empty token
                result = loop.run_until_complete(api_test_alert())
                self.assertEqual(result["status"], "ok")
                self.assertFalse(result["telegram_sent"])
            finally:
                loop.close()

    def test_07_token_sanitization_in_logs(self):
        """Ensure exceptions and error logs never expose the Telegram Bot Token."""
        secret_token = "8867075575:AAFy61PGTllqLewQaJ2Upe0ytAnQH2IGIZE"
        from app.backend.main import _sanitize_log

        leak_message = f"Failed to connect to https://api.telegram.org/bot{secret_token}/sendMessage"
        sanitized = _sanitize_log(leak_message, secret_token)

        self.assertNotIn(secret_token, sanitized)
        self.assertIn("[REDACTED]", sanitized)

    def test_08_local_time_formatting(self):
        """Verify local_time string matches requested '17 Oct 2026, 6:22 PM' format."""
        formatted = get_local_time_str()
        self.assertTrue(len(formatted) > 5)
        # Should contain comma and AM/PM
        self.assertIn(",", formatted)
        self.assertTrue("AM" in formatted or "PM" in formatted)


if __name__ == "__main__":
    unittest.main()
