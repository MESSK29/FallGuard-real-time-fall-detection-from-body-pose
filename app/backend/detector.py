"""detector.py - Real-Time Fall Detection State Machine and Feature Pipeline.

One FallDetector instance is maintained per active WebSocket connection.
Reuses src.features for exact causal kinematics, rolling floor lookback, and
window statistical features.

Pipeline:
1. Buffers raw incoming frames over a 6-second lookback window.
2. Resamples buffered landmarks onto a uniform 30 Hz grid (bounded linear interpolation,
   gaps > 0.3s left as NaN).
3. Applies 3-frame centered moving average smoothing on (x, y, z) coordinates.
4. Computes per-frame measurements (src.features.frame_measurements) and extracts
   the active 2.0-second sliding window features (src.features.live_window_features).
5. Imputes NaNs using training-set medians and scores probability using models/{MODEL_NAME}.joblib.
6. Finite State Machine:
   NORMAL -> POSSIBLE_FALL (prob >= threshold) -> ALERT (post-fall lying confirmation
   held for LYING_SECONDS). 60-second cooldown after alert.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.features import (
    IMAGE_H,
    IMAGE_W,
    MIN_VALID_FRACTION,
    WINDOW_S,
    frame_measurements,
    live_window_features,
)

load_dotenv()
logger = logging.getLogger("fallguard.detector")

MODELS_DIR = PROJECT_ROOT / "models"

# Global cached model bundles so files are loaded once at startup
_MODEL_BUNDLES: Dict[str, Dict[str, Any]] = {}


def load_model_bundle(model_name: Optional[str] = None) -> Dict[str, Any]:
    """Load and cache model bundle from models/{model_name}.joblib."""
    if model_name is None:
        model_name = os.getenv("MODEL_NAME", "xgb").lower()

    norm_name = model_name.lower().replace("-", "_")
    if norm_name in ["xgb", "xgboost"]:
        key = "xgb"
    elif norm_name in ["rf", "random_forest"]:
        key = "rf"
    else:
        key = norm_name

    if key in _MODEL_BUNDLES:
        return _MODEL_BUNDLES[key]

    target_file = MODELS_DIR / f"{key}.joblib"
    if not target_file.exists():
        # Fallback to xgb.joblib
        target_file = MODELS_DIR / "xgb.joblib"
        if not target_file.exists():
            raise FileNotFoundError(f"Model bundle not found at {target_file}")
        key = "xgb"

    logger.info(f"Loading FallDetector model bundle: {target_file}")
    bundle = joblib.load(target_file)
    _MODEL_BUNDLES[key] = bundle
    return bundle


class FallDetector:
    """Real-time fall detection state machine for live telemetry streams."""

    BUFFER_LOOKBACK_S: float = 6.0       # Retain 6s of raw frames for causal floor & window
    GRID_FPS: float = 30.0               # Resample incoming frames onto 30 Hz uniform grid
    MAX_INTERPOLATION_GAP_S: float = 0.3 # Gaps longer than 0.3s remain un-interpolated (NaN)
    INFERENCE_STRIDE_S: float = 0.25     # Evaluate window model every 0.25 seconds
    POSSIBLE_FALL_TIMEOUT_S: float = 10.0# Reset to normal if not confirmed within 10s
    COOLDOWN_SECONDS: float = 60.0       # Cooldown after confirmed alert
    TRACKING_LOST_THRESHOLD_S: float = 1.0# Mark missing if no frame for >= 1.0s
    LOST_TRACKING_LYING_WINDOW_S: float = 3.0 # Count as still lying if lost <= 3s post-fall

    def __init__(
        self,
        model_name: Optional[str] = None,
        probability_threshold: Optional[float] = None,
        lying_seconds: Optional[float] = None,
    ):
        self.bundle = load_model_bundle(model_name)
        self.model = self.bundle["model"]
        self.feature_names: List[str] = self.bundle["feature_names"]
        self.medians: Dict[str, float] = self.bundle["medians"]

        self.saved_threshold: float = float(self.bundle.get("threshold", 0.65))

        # Probability threshold: env override or bundle default
        if probability_threshold is not None:
            self.threshold = float(probability_threshold)
        else:
            env_thresh = os.getenv("PROBABILITY_THRESHOLD")
            if env_thresh is not None and env_thresh.strip():
                self.threshold = float(env_thresh)
            else:
                self.threshold = self.saved_threshold

        # Lying confirmation duration
        if lying_seconds is not None:
            self.lying_seconds = float(lying_seconds)
        else:
            env_lying = os.getenv("LYING_SECONDS")
            self.lying_seconds = float(env_lying) if env_lying else 2.0

        # Raw frame buffer: list of (t_sec, np.ndarray shape (33, 4))
        self.raw_buffer: List[Tuple[float, np.ndarray]] = []

        # Internal State Machine
        self.status: str = "normal"      # "normal" | "possible_fall" | "alert"
        self.current_probability: float = 0.0
        self.peak_probability: float = 0.0
        self.active_alert_id: Optional[int] = None
        self.lying_progress: float = 0.0
        self.last_inference_t: Optional[float] = None

        # Timing tracking
        self.last_frame_t: Optional[float] = None
        self.last_valid_tracking_t: Optional[float] = None
        self.possible_fall_start_t: Optional[float] = None
        self.lying_accumulated_seconds: float = 0.0
        self.last_state_eval_t: Optional[float] = None
        self.cooldown_until_t: float = 0.0

        # Latest window measurements
        self.last_torso_angle_mean: float = np.nan
        self.last_hip_height_mean: float = np.nan
        self.latest_window_features: Optional[Dict[str, float]] = None
        self.computed_windows: List[Dict[str, Any]] = []

    def add_frame(self, t_ms: float, landmarks: List[List[float]]) -> Dict[str, Any]:
        """Ingest one incoming video frame with 33 landmarks [x, y, z, visibility].

        Args:
            t_ms: Frame timestamp in milliseconds.
            landmarks: Array of 33 points, each with [x, y, z, visibility].

        Returns:
            Dict {"status": "normal|possible_fall|alert", "probability": float, "lying_progress": float}
        """
        t_sec = float(t_ms) / 1000.0
        lm_arr = np.asarray(landmarks, dtype=float)

        # Detect time gaps (if no frame arrived for >= 1s, person was missing)
        person_was_missing = False
        if self.last_frame_t is not None and (t_sec - self.last_frame_t) >= self.TRACKING_LOST_THRESHOLD_S:
            person_was_missing = True

        self.last_frame_t = t_sec
        self.last_valid_tracking_t = t_sec

        # Append to raw buffer
        self.raw_buffer.append((t_sec, lm_arr))

        # Prune buffer to keep only the last BUFFER_LOOKBACK_S (6.0s)
        cutoff_t = t_sec - self.BUFFER_LOOKBACK_S
        self.raw_buffer = [(t, lm) for t, lm in self.raw_buffer if t >= cutoff_t]

        # Check if enough temporal history exists (at least 2.0s)
        buffer_duration = self.raw_buffer[-1][0] - self.raw_buffer[0][0]
        if buffer_duration < WINDOW_S:
            self._update_state_machine(t_sec, new_inference=False)
            return self._build_response()

        # Execute inference every INFERENCE_STRIDE_S (0.25s)
        should_infer = (
            self.last_inference_t is None
            or (t_sec - self.last_inference_t) >= (self.INFERENCE_STRIDE_S - 1e-4)
        )

        if should_infer:
            self._run_inference(t_sec)

        self._update_state_machine(t_sec, new_inference=should_infer)
        return self._build_response()

    def check_timeout(self, current_t_ms: Optional[float] = None) -> Dict[str, Any]:
        """Called periodically or when no frames arrive to handle missing person and timeouts."""
        if current_t_ms is not None:
            t_sec = float(current_t_ms) / 1000.0
        else:
            t_sec = time.time()

        # If no frame for >= 1s, person is missing
        person_missing = (
            self.last_frame_t is not None
            and (t_sec - self.last_frame_t) >= self.TRACKING_LOST_THRESHOLD_S
        )

        self._update_state_machine(t_sec, new_inference=False, person_missing=person_missing)
        return self._build_response()

    def _run_inference(self, t_curr: float) -> None:
        """Resample raw buffer onto 30 Hz grid, smooth, extract window features, and score ML model."""
        self.last_inference_t = t_curr

        # 1. Resample onto uniform 30 Hz time grid
        raw_times = np.array([item[0] for item in self.raw_buffer], dtype=float)
        raw_lms = [item[1] for item in self.raw_buffer]

        t_start = raw_times[0]
        t_end = raw_times[-1]
        dt = 1.0 / self.GRID_FPS

        grid_times = np.arange(t_start, t_end + 1e-6, dt)
        n_grid = len(grid_times)
        if n_grid < int(WINDOW_S * self.GRID_FPS * 0.8):
            return

        resampled_lms = np.full((n_grid, 33, 4), np.nan, dtype=float)

        for g_idx, tg in enumerate(grid_times):
            idx = np.searchsorted(raw_times, tg)
            if idx == 0:
                dt_gap = raw_times[0] - tg
                if dt_gap <= 0.05:
                    resampled_lms[g_idx] = raw_lms[0]
            elif idx >= len(raw_times):
                dt_gap = tg - raw_times[-1]
                if dt_gap <= 0.05:
                    resampled_lms[g_idx] = raw_lms[-1]
            else:
                t_prev = raw_times[idx - 1]
                t_next = raw_times[idx]
                gap = t_next - t_prev
                if gap <= self.MAX_INTERPOLATION_GAP_S:
                    alpha = (tg - t_prev) / gap
                    resampled_lms[g_idx] = (1.0 - alpha) * raw_lms[idx - 1] + alpha * raw_lms[idx]

        # 2. Build DataFrame for src.features
        data: Dict[str, Any] = {"time_s": grid_times}
        for i in range(33):
            data[f"lm{i}_x"] = resampled_lms[:, i, 0]
            data[f"lm{i}_y"] = resampled_lms[:, i, 1]
            data[f"lm{i}_z"] = resampled_lms[:, i, 2]
            data[f"lm{i}_v"] = resampled_lms[:, i, 3]

        df_grid = pd.DataFrame(data)

        # 3. 3-Frame centered moving average smoothing on (x, y, z) coordinates only
        coord_cols = [f"lm{i}_{c}" for i in range(33) for c in ["x", "y", "z"]]
        df_grid[coord_cols] = df_grid[coord_cols].rolling(window=3, center=True, min_periods=1).mean()

        # 4. Compute per-frame causal measurements (torso angle, hip height above floor, speed, aspect)
        measurements = frame_measurements(df_grid)

        # 5. Compute live window features on the active 2.0s window
        feats = live_window_features(measurements)

        if feats is not None:
            self.latest_window_features = feats.copy()
            w_end = round(float(measurements["time_s"].iloc[-1]), 4)
            w_start = round(w_end - WINDOW_S, 4)
            w_record = feats.copy()
            w_record["window_start"] = w_start
            w_record["window_end"] = w_end
            self.computed_windows.append(w_record)

            # Impute NaNs with training medians
            feature_vector = [
                float(feats[col]) if (col in feats and not np.isnan(feats[col])) else float(self.medians.get(col, 0.0))
                for col in self.feature_names
            ]

            X = np.array([feature_vector], dtype=float)
            probs = self.model.predict_proba(X)
            self.current_probability = float(probs[0, 1])
            if self.status == "possible_fall":
                self.peak_probability = max(self.peak_probability, self.current_probability)

            self.last_torso_angle_mean = float(feats.get("torso_angle_mean", np.nan))
            self.last_hip_height_mean = float(feats.get("hip_height_mean", np.nan))
        else:
            # Valid fraction < 0.6 or not enough valid torso frames
            self.last_torso_angle_mean = np.nan
            self.last_hip_height_mean = np.nan

    def _update_state_machine(
        self,
        t_curr: float,
        new_inference: bool = False,
        person_missing: bool = False,
    ) -> None:
        """Finite State Machine handling transitions and post-fall confirmation."""
        # Calculate time step for accumulator
        if self.last_state_eval_t is not None:
            dt = max(0.0, min(t_curr - self.last_state_eval_t, 0.5))
        else:
            dt = 0.0
        self.last_state_eval_t = t_curr

        # ---------------------------------------------------------------------
        # 1. Active Cooldown Check
        # ---------------------------------------------------------------------
        if t_curr < self.cooldown_until_t:
            # Inside 60s cooldown: suppress alerts and possible falls
            self.status = "normal"
            self.lying_progress = 0.0
            return
        elif self.cooldown_until_t > 0.0:
            # Cooldown expired: clear active alert
            self.active_alert_id = None

        # ---------------------------------------------------------------------
        # 2. State: NORMAL
        # ---------------------------------------------------------------------
        if self.status == "normal":
            self.lying_progress = 0.0
            self.lying_accumulated_seconds = 0.0

            # Missing person or low visibility without prior possible fall must NEVER alert
            if person_missing:
                self.current_probability = 0.0
                return

            if new_inference and self.current_probability >= self.threshold:
                # Trigger Candidate Fall!
                self.status = "possible_fall"
                self.possible_fall_start_t = t_curr
                self.lying_accumulated_seconds = 0.0
                self.peak_probability = self.current_probability
                logger.info(
                    f"Candidate Fall Triggered! Prob={self.current_probability:.3f} >= Thresh={self.threshold:.3f}. "
                    f"Waiting for lying confirmation ({self.lying_seconds}s)..."
                )

        # ---------------------------------------------------------------------
        # 3. State: POSSIBLE_FALL
        # ---------------------------------------------------------------------
        elif self.status == "possible_fall":
            if new_inference:
                self.peak_probability = max(self.peak_probability, self.current_probability)

            time_in_possible_fall = (
                t_curr - self.possible_fall_start_t if self.possible_fall_start_t is not None else 0.0
            )

            # Rule A: Timeout after 10s without confirmation
            if time_in_possible_fall > self.POSSIBLE_FALL_TIMEOUT_S:
                logger.info("Possible fall timed out (10s elapsed without confirmation). Returning to normal.")
                self.status = "normal"
                self.lying_progress = 0.0
                self.lying_accumulated_seconds = 0.0
                self.peak_probability = 0.0
                return

            # Rule B: Person is clearly upright again -> False Alarm / Recovery
            if (
                not np.isnan(self.last_torso_angle_mean)
                and self.last_torso_angle_mean < 30.0
                and self.last_hip_height_mean > 0.8
            ):
                logger.info(
                    f"Person clearly upright again (angle={self.last_torso_angle_mean:.1f}°, "
                    f"height={self.last_hip_height_mean:.2f}). Returning to normal."
                )
                self.status = "normal"
                self.lying_progress = 0.0
                self.lying_accumulated_seconds = 0.0
                self.peak_probability = 0.0
                return

            # Rule C: Lying-Still Confirmation Condition
            # Condition 1: Visible lying posture (angle > 60° AND hip height < 0.5)
            is_visibly_lying = (
                not np.isnan(self.last_torso_angle_mean)
                and self.last_torso_angle_mean > 60.0
                and self.last_hip_height_mean < 0.5
            )

            # Condition 2: Tracking lost right after possible fall (person on floor out of camera view)
            time_since_tracking_lost = (
                t_curr - self.last_valid_tracking_t if self.last_valid_tracking_t is not None else 0.0
            )
            is_lost_on_floor = (
                (person_missing or np.isnan(self.last_torso_angle_mean))
                and time_in_possible_fall <= self.LOST_TRACKING_LYING_WINDOW_S
            )

            if is_visibly_lying or is_lost_on_floor:
                self.lying_accumulated_seconds += dt
                self.lying_progress = min(1.0, self.lying_accumulated_seconds / self.lying_seconds)

                if self.lying_accumulated_seconds >= self.lying_seconds:
                    # CONFIRMED FALL ALERT!
                    self.status = "alert"
                    self.cooldown_until_t = t_curr + self.COOLDOWN_SECONDS
                    self.lying_progress = 1.0
                    logger.warning(
                        f"*** FALL CONFIRMED! Alert raised at t={t_curr:.2f}s (peak prob={self.peak_probability:.3f}). "
                        f"Cooldown active for {self.COOLDOWN_SECONDS}s. ***"
                    )
            else:
                # Posture is ambiguous; progress remains unchanged
                self.lying_progress = min(1.0, self.lying_accumulated_seconds / self.lying_seconds)

        # ---------------------------------------------------------------------
        # 4. State: ALERT
        # ---------------------------------------------------------------------
        elif self.status == "alert":
            # Once alert is reported, transition to normal under cooldown
            self.status = "normal"
            self.lying_progress = 0.0
            self.lying_accumulated_seconds = 0.0

    def _build_response(self) -> Dict[str, Any]:
        """Construct standard response payload for WebSocket client."""
        prob = self.peak_probability if self.status == "alert" else self.current_probability
        payload: Dict[str, Any] = {
            "status": self.status,
            "probability": round(float(prob), 4),
            "lying_progress": round(float(self.lying_progress), 3),
        }
        if self.status == "alert":
            payload["peak_probability"] = round(float(self.peak_probability), 4)
        if self.active_alert_id is not None:
            payload["alert_id"] = self.active_alert_id
        return payload

    def apply_settings(self, settings: Any) -> Dict[str, Any]:
        """Apply and clamp per-connection client settings.

        Supported fields:
        - sensitivity: 1 to 5 (default 3 = saved threshold). Higher sensitivity lowers threshold.
        - lying_seconds: 1.0 to 5.0 (default 2.0).
        """
        if not isinstance(settings, dict):
            return {
                "threshold": self.threshold,
                "lying_seconds": self.lying_seconds,
            }

        # Sensitivity: 1 to 5
        raw_sens = settings.get("sensitivity")
        if raw_sens is not None:
            try:
                sens = int(round(float(raw_sens)))
                sens = max(1, min(5, sens))
                # Mapping: 1 -> +0.20, 2 -> +0.10, 3 -> 0.0, 4 -> -0.10, 5 -> -0.20
                offset = (3 - sens) * 0.10
                base = self.saved_threshold
                self.threshold = round(max(0.30, min(0.95, base + offset)), 4)
            except (ValueError, TypeError):
                pass

        # Lying delay: 1.0 to 5.0s
        raw_lying = settings.get("lying_seconds")
        if raw_lying is not None:
            try:
                lying = float(raw_lying)
                self.lying_seconds = round(max(1.0, min(5.0, lying)), 2)
            except (ValueError, TypeError):
                pass

        return {
            "threshold": self.threshold,
            "lying_seconds": self.lying_seconds,
        }

