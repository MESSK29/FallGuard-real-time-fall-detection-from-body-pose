"""test_incident_clip.py - Test suite for Opt-In Incident Video Clip Processing and Delivery.

Meets Requirement 8:
- Telegram and ffmpeg mocked (no real network calls)
- Consent required (403)
- Wrong client rejected (404)
- Size limits (413) and type limits (415)
- One clip per alert (409)
- Dismissed alert skipped (409)
- Destination chat_id equals the chat used for the alert text
- Temp directory removed even on error
- /alerts never exposes the full chat_id
- One real-ffmpeg test that joins two tiny generated segments (skipped if ffmpeg is not installed).
"""

import asyncio
import io
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from starlette.testclient import TestClient

from app.backend.incident_clip import (
    get_ffmpeg_path,
    is_ffmpeg_available,
    mask_chat_id,
    process_incident_clip,
)
from app.backend.main import (
    app,
    create_alert,
    in_memory_alerts,
    clip_consent,
    client_clip_consents,
    client_clip_timestamps,
    client_test_clip_timestamps,
    client_verified_chat_ids,
)


@pytest.fixture(autouse=True)
def reset_incident_state():
    """Reset all alerts, consents, timestamps, and caches before and after each test."""
    in_memory_alerts.clear()
    clip_consent.clear()
    client_clip_timestamps.clear()
    client_test_clip_timestamps.clear()
    client_verified_chat_ids.clear()
    yield
    in_memory_alerts.clear()
    clip_consent.clear()
    client_clip_timestamps.clear()
    client_test_clip_timestamps.clear()
    client_verified_chat_ids.clear()


@pytest.fixture
def client():
    """Starlette TestClient instance."""
    return TestClient(app)


def make_dummy_segment(size_bytes: int = 1024, name: str = "seg.webm", content_type: str = "video/webm"):
    """Helper to create dummy in-memory file for multipart upload."""
    return (name, io.BytesIO(b"x" * size_bytes), content_type)


class TestIncidentClipValidation:
    """Tests for consent, authorization, rate limits, and payload restrictions."""

    def test_consent_required_403(self, client):
        """Upload without client consent must be rejected with HTTP 403."""
        cid = "client-no-consent"
        clip_consent[cid] = False

        alert = create_alert(
            probability=0.92,
            client_id=cid,
            destination_chat_id="12345678",
            clip_status="pending",
        )

        files = [("segments", make_dummy_segment())]
        metadata = json.dumps({"T": 1000.0, "segments": [{"start_ms": 500.0}]})

        resp = client.post(
            f"/alerts/{alert['id']}/clip",
            headers={"X-Client-Id": cid},
            files=files,
            data={"metadata": metadata},
        )
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Turn on 'Send video clip' and tick the consent box, then try again."

    def test_test_alert_without_consent_rejected_403(self, client):
        """Test clip authorization without consent must be rejected with HTTP 403."""
        cid = "client-test-no-consent"
        clip_consent[cid] = False

        resp = client.post(
            "/test-alert",
            headers={"X-Client-Id": cid},
            json={"is_test_clip": True},
        )
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Turn on 'Send video clip' and tick the consent box, then try again."

    def test_consent_toggle_sequence_on_off_on(self, client):
        """Toggling consent on, off, on ends as True in clip_consent."""
        cid = "client-toggle-seq"

        # 1. Toggle ON
        r1 = client.post("/api/consent/clip", headers={"X-Client-Id": cid}, json={"consent": True})
        assert r1.status_code == 200
        assert r1.json()["consent"] is True
        assert clip_consent[cid] is True

        # 2. Toggle OFF
        r2 = client.post("/api/consent/clip", headers={"X-Client-Id": cid}, json={"consent": False})
        assert r2.status_code == 200
        assert r2.json()["consent"] is False
        assert clip_consent[cid] is False

        # 3. Toggle ON
        r3 = client.post("/api/consent/clip", headers={"X-Client-Id": cid}, json={"consent": True})
        assert r3.status_code == 200
        assert r3.json()["consent"] is True
        assert clip_consent[cid] is True

    def test_consent_reject_missing_or_non_boolean_422(self, client):
        """Missing or non-boolean consent field must be rejected with HTTP 422."""
        cid = "client-bad-consent"

        # Missing "consent" field
        r1 = client.post("/api/consent/clip", headers={"X-Client-Id": cid}, json={})
        assert r1.status_code == 422

        # Non-boolean string
        r2 = client.post("/api/consent/clip", headers={"X-Client-Id": cid}, json={"consent": "true"})
        assert r2.status_code == 422

        # Non-boolean integer
        r3 = client.post("/api/consent/clip", headers={"X-Client-Id": cid}, json={"consent": 1})
        assert r3.status_code == 422

        # Non-boolean None
        r4 = client.post("/api/consent/clip", headers={"X-Client-Id": cid}, json={"consent": None})
        assert r4.status_code == 422

    def test_ws_and_http_cannot_disagree(self, client):
        """Both WebSocket settings and POST /api/consent/clip write to the single clip_consent dict."""
        cid = "client-ws-http-sync"

        # Set True via HTTP
        r1 = client.post("/api/consent/clip", headers={"X-Client-Id": cid}, json={"consent": True})
        assert r1.status_code == 200
        assert clip_consent[cid] is True

        # Connect WebSocket and send settings with consent
        with client.websocket_connect(f"/ws?client_id={cid}") as ws:
            ws.send_json({"type": "settings", "consent": False})
            ack = ws.receive_json()
            assert ack.get("type") == "settings_ack"
            assert ack.get("consent") is False
            assert clip_consent[cid] is False

            # Switch back to True over WS
            ws.send_json({"type": "settings", "consent": True})
            ack2 = ws.receive_json()
            assert ack2.get("type") == "settings_ack"
            assert ack2.get("consent") is True
            assert clip_consent[cid] is True

    def test_simulated_server_restart_resync(self, client):
        """A simulated server restart (clearing in-memory state) re-syncs from saved client consent."""
        cid = "client-restart-resync"

        # Initially set consent to True
        client.post("/api/consent/clip", headers={"X-Client-Id": cid}, json={"consent": True})
        assert clip_consent[cid] is True

        # Simulate server restart: in-memory state is cleared
        clip_consent.clear()
        assert cid not in clip_consent

        # Client re-connects on page load / reconnect and sends saved consent
        r = client.post("/api/consent/clip", headers={"X-Client-Id": cid}, json={"consent": True})
        assert r.status_code == 200
        assert r.json()["consent"] is True
        assert clip_consent[cid] is True

    def test_wrong_client_rejected_404(self, client):
        """Uploads from a client that does not own the alert must return HTTP 404."""
        owner_cid = "client-owner"
        other_cid = "client-imposter"

        client_clip_consents[owner_cid] = True
        client_clip_consents[other_cid] = True

        alert = create_alert(
            probability=0.88,
            client_id=owner_cid,
            destination_chat_id="12345678",
            clip_status="pending",
        )

        files = [("segments", make_dummy_segment())]
        metadata = json.dumps({"T": 1000.0, "segments": [{"start_ms": 500.0}]})

        resp = client.post(
            f"/alerts/{alert['id']}/clip",
            headers={"X-Client-Id": other_cid},
            files=files,
            data={"metadata": metadata},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_segment_size_limit_413(self, client):
        """Individual segments larger than 3 MB must return HTTP 413."""
        cid = "client-big-seg"
        client_clip_consents[cid] = True

        alert = create_alert(
            probability=0.90,
            client_id=cid,
            destination_chat_id="12345678",
            clip_status="pending",
        )

        # 3.2 MB dummy segment
        big_seg = make_dummy_segment(size_bytes=int(3.2 * 1024 * 1024))
        files = [("segments", big_seg)]
        metadata = json.dumps({"T": 1000.0, "segments": [{"start_ms": 500.0}]})

        resp = client.post(
            f"/alerts/{alert['id']}/clip",
            headers={"X-Client-Id": cid},
            files=files,
            data={"metadata": metadata},
        )
        assert resp.status_code == 413
        assert "exceeds maximum 3 mb" in resp.json()["detail"].lower()

    def test_total_payload_size_limit_413(self, client):
        """Total payload larger than 15 MB must return HTTP 413."""
        cid = "client-total-payload"
        client_clip_consents[cid] = True

        alert = create_alert(
            probability=0.90,
            client_id=cid,
            destination_chat_id="12345678",
            clip_status="pending",
        )

        # 6 segments of 2.7 MB each = 16.2 MB > 15 MB
        files = [("segments", make_dummy_segment(size_bytes=int(2.7 * 1024 * 1024), name=f"s{i}.webm")) for i in range(6)]
        metadata = json.dumps({"T": 1000.0, "segments": [{"start_ms": 500.0 * i} for i in range(6)]})

        resp = client.post(
            f"/alerts/{alert['id']}/clip",
            headers={"X-Client-Id": cid},
            files=files,
            data={"metadata": metadata},
        )
        assert resp.status_code == 413
        assert "15 mb limit" in resp.json()["detail"].lower()

    def test_maximum_segments_count_413(self, client):
        """More than 14 segments must return HTTP 413."""
        cid = "client-many-segs"
        client_clip_consents[cid] = True

        alert = create_alert(
            probability=0.90,
            client_id=cid,
            destination_chat_id="12345678",
            clip_status="pending",
        )

        # 15 segments
        files = [("segments", make_dummy_segment(size_bytes=100, name=f"s{i}.webm")) for i in range(15)]
        metadata = json.dumps({"T": 1000.0, "segments": [{"start_ms": 100.0 * i} for i in range(15)]})

        resp = client.post(
            f"/alerts/{alert['id']}/clip",
            headers={"X-Client-Id": cid},
            files=files,
            data={"metadata": metadata},
        )
        assert resp.status_code == 413
        assert "maximum 14 allowed" in resp.json()["detail"].lower()

    def test_unsupported_media_type_415(self, client):
        """Non webm/mp4 content types must return HTTP 415."""
        cid = "client-bad-type"
        client_clip_consents[cid] = True

        alert = create_alert(
            probability=0.90,
            client_id=cid,
            destination_chat_id="12345678",
            clip_status="pending",
        )

        files = [("segments", make_dummy_segment(content_type="image/jpeg", name="frame.jpg"))]
        metadata = json.dumps({"T": 1000.0, "segments": [{"start_ms": 500.0}]})

        resp = client.post(
            f"/alerts/{alert['id']}/clip",
            headers={"X-Client-Id": cid},
            files=files,
            data={"metadata": metadata},
        )
        assert resp.status_code == 415
        assert "unsupported media type" in resp.json()["detail"].lower()

    def test_one_clip_per_alert_409(self, client):
        """Only one clip can be sent per alert. A second attempt must return HTTP 409."""
        cid = "client-duplicate-clip"
        client_clip_consents[cid] = True

        alert = create_alert(
            probability=0.95,
            client_id=cid,
            destination_chat_id="12345678",
            clip_status="sent",
        )

        files = [("segments", make_dummy_segment())]
        metadata = json.dumps({"T": 1000.0, "segments": [{"start_ms": 500.0}]})

        resp = client.post(
            f"/alerts/{alert['id']}/clip",
            headers={"X-Client-Id": cid},
            files=files,
            data={"metadata": metadata},
        )
        assert resp.status_code == 409
        assert "already processed" in resp.json()["detail"].lower()

    def test_dismissed_alert_skipped_409(self, client):
        """Uploading for a dismissed alert must be skipped with HTTP 409 and mark clip_status skipped_dismissed."""
        cid = "client-dismissed"
        client_clip_consents[cid] = True

        alert = create_alert(
            probability=0.91,
            client_id=cid,
            destination_chat_id="12345678",
            clip_status="pending",
        )
        alert["status"] = "dismissed"

        files = [("segments", make_dummy_segment())]
        metadata = json.dumps({"T": 1000.0, "segments": [{"start_ms": 500.0}]})

        resp = client.post(
            f"/alerts/{alert['id']}/clip",
            headers={"X-Client-Id": cid},
            files=files,
            data={"metadata": metadata},
        )
        assert resp.status_code == 409
        assert "dismissed" in resp.json()["detail"].lower()
        assert alert["clip_status"] == "skipped_dismissed"


class TestIncidentClipDelivery:
    """Tests for Telegram dispatch destination, masking, and temporary directory cleanup."""

    @patch("app.backend.main.process_incident_clip", new_callable=AsyncMock)
    @patch("app.backend.main.send_telegram_video", new_callable=AsyncMock)
    def test_destination_chat_id_equals_alert_chat(self, mock_send_video, mock_process, client):
        """The incident clip must be dispatched to the EXACT chat_id used for the alert text."""
        cid = "client-dest-check"
        client_clip_consents[cid] = True
        custom_chat = "9988776655"

        mock_process.return_value = None
        mock_send_video.return_value = True

        alert = create_alert(
            probability=0.96,
            client_id=cid,
            destination_chat_id=custom_chat,
            clip_status="pending",
        )

        files = [("segments", make_dummy_segment())]
        metadata = json.dumps({"T": 1000.0, "segments": [{"start_ms": 500.0}]})

        resp = client.post(
            f"/alerts/{alert['id']}/clip",
            headers={"X-Client-Id": cid},
            files=files,
            data={"metadata": metadata},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert alert["clip_status"] == "sent"

        # Verify send_telegram_video was invoked with the exact stored destination_chat_id
        mock_send_video.assert_called_once()
        call_kwargs = mock_send_video.call_args.kwargs
        assert call_kwargs.get("chat_id") == custom_chat

    @patch("app.backend.main.process_incident_clip", new_callable=AsyncMock)
    def test_temp_directory_removed_even_on_error(self, mock_process, client):
        """The temporary processing directory must be cleaned up in all cases, even on errors."""
        cid = "client-temp-clean"
        client_clip_consents[cid] = True

        captured_tmp_dirs = []

        async def fake_failing_process(segment_paths, segment_start_times, T_ms, output_path):
            dir_path = os.path.dirname(output_path)
            captured_tmp_dirs.append(dir_path)
            # Verify directory exists while executing
            assert os.path.isdir(dir_path)
            raise RuntimeError("Simulated ffmpeg failure during encoding")

        mock_process.side_effect = fake_failing_process

        alert = create_alert(
            probability=0.93,
            client_id=cid,
            destination_chat_id="12345678",
            clip_status="pending",
        )

        files = [("segments", make_dummy_segment())]
        metadata = json.dumps({"T": 1000.0, "segments": [{"start_ms": 500.0}]})

        resp = client.post(
            f"/alerts/{alert['id']}/clip",
            headers={"X-Client-Id": cid},
            files=files,
            data={"metadata": metadata},
        )
        assert resp.status_code == 500
        assert "Simulated ffmpeg failure" in resp.json()["detail"]

        # Verify that the captured temp directory was completely removed by the finally block
        assert len(captured_tmp_dirs) == 1
        tmp_dir = captured_tmp_dirs[0]
        assert not os.path.exists(tmp_dir), f"Temp directory leaked: {tmp_dir}"

    def test_alerts_never_exposes_full_chat_id(self, client):
        """GET /alerts must always mask the destination_chat_id and never expose the raw chat ID."""
        cid = "client-privacy-mask"
        secret_chat_id = "5513741585"

        create_alert(
            probability=0.89,
            client_id=cid,
            destination_chat_id=secret_chat_id,
            clip_status="sent",
        )

        resp = client.get("/alerts", headers={"X-Client-Id": cid})
        assert resp.status_code == 200

        data = resp.json()
        assert len(data) == 1
        alert_record = data[0]

        # Verify masked display (last 4 digits only)
        assert alert_record["destination_chat_id"] == "...1585"
        assert alert_record.get("chat_id_masked") == "...1585"

        # Assert raw unmasked chat ID is never returned anywhere in response text
        assert secret_chat_id not in resp.text


class TestRealFfmpegIntegration:
    """Real ffmpeg test joining two tiny generated segments."""

    def test_real_ffmpeg_join_two_generated_segments(self):
        """Generates two tiny 1-second video segments, joins with process_incident_clip, and validates output."""
        if not is_ffmpeg_available():
            pytest.skip("ffmpeg executable is not installed on this system")

        ffmpeg_bin = get_ffmpeg_path()
        assert ffmpeg_bin is not None and os.path.isfile(ffmpeg_bin)

        with tempfile.TemporaryDirectory() as work_dir:
            seg1 = os.path.join(work_dir, "seg1.mp4")
            seg2 = os.path.join(work_dir, "seg2.mp4")
            out_clip = os.path.join(work_dir, "output.mp4")

            # Generate two 1-second valid test videos using ffmpeg testsrc
            cmd1 = [
                ffmpeg_bin, "-y", "-f", "lavfi",
                "-i", "testsrc=duration=1:size=320x240:rate=10",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                seg1
            ]
            cmd2 = [
                ffmpeg_bin, "-y", "-f", "lavfi",
                "-i", "testsrc=duration=1:size=320x240:rate=10",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                seg2
            ]

            proc1 = subprocess.run(cmd1, capture_output=True)
            assert proc1.returncode == 0, f"Failed generating seg1: {proc1.stderr.decode()}"
            proc2 = subprocess.run(cmd2, capture_output=True)
            assert proc2.returncode == 0, f"Failed generating seg2: {proc2.stderr.decode()}"

            assert os.path.isfile(seg1) and os.path.getsize(seg1) > 0
            assert os.path.isfile(seg2) and os.path.getsize(seg2) > 0

            # Execute real incident clip processing: join segments, trim to [T-10s, T+10s]
            start_times = [1000.0, 2000.0]
            T_ms = 1500.0

            asyncio.run(
                process_incident_clip(
                    segment_paths=[seg1, seg2],
                    segment_start_times=start_times,
                    T_ms=T_ms,
                    output_path=out_clip,
                )
            )

            # Validate processed output
            assert os.path.isfile(out_clip)
            out_size = os.path.getsize(out_clip)
            assert out_size > 0
            assert out_size < 20 * 1024 * 1024  # strictly under 20 MB ceiling
