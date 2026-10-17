"""features.py - Single Source of Truth for Feature Engineering.

This module is strictly shared between offline model training and the real-time
backend inference pipeline. All calculations are strictly causal (using only current
and past frames) to ensure 100% parity and zero temporal leakage.

Constants:
----------
IMAGE_W = 640
IMAGE_H = 480
WINDOW_S = 2.0
STRIDE_S = 0.25
MIN_VISIBILITY = 0.5
MIN_VALID_FRACTION = 0.6
FALL_COVERAGE = 0.5
FLOOR_LOOKBACK_S = 3.0
SPEED_LOOKBACK_S = 0.2
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ============================================================================
# Global Configuration Constants
# ============================================================================
IMAGE_W: int = 640
IMAGE_H: int = 480
WINDOW_S: float = 2.0
STRIDE_S: float = 0.25
MIN_VISIBILITY: float = 0.5
MIN_VALID_FRACTION: float = 0.6
FALL_COVERAGE: float = 0.5
FLOOR_LOOKBACK_S: float = 3.0
SPEED_LOOKBACK_S: float = 0.2

# Landmark indices (MediaPipe Pose)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

MEASUREMENT_COLS: List[str] = ["torso_angle", "hip_height", "hip_speed", "aspect_ratio"]

FEATURE_GROUPS: Dict[str, List[str]] = {
    "angle": [
        "torso_angle_min",
        "torso_angle_max",
        "torso_angle_mean",
        "torso_angle_std",
        "torso_angle_change",
    ],
    "hip_height": [
        "hip_height_min",
        "hip_height_max",
        "hip_height_mean",
        "hip_height_std",
        "hip_height_change",
    ],
    "speed": [
        "hip_speed_min",
        "hip_speed_max",
        "hip_speed_mean",
        "hip_speed_std",
        "hip_speed_change",
    ],
    "aspect": [
        "aspect_ratio_min",
        "aspect_ratio_max",
        "aspect_ratio_mean",
        "aspect_ratio_std",
        "aspect_ratio_change",
    ],
}

FEATURE_COLS: List[str] = [col for group in FEATURE_GROUPS.values() for col in group]


# ============================================================================
# 1. Per-Frame Measurements (Causal)
# ============================================================================

def frame_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Compute normalized geometric and velocity measurements for each frame.

    Args:
        df: DataFrame containing landmark columns lm{i}_x, lm{i}_y, lm{i}_z, lm{i}_v
            for i in 0..32, plus time_s (in seconds).

    Returns:
        DataFrame with columns:
        - time_s
        - torso_len: reference causal rolling median torso length
        - torso_angle: folded angle [0, 90] from vertical using atan2(abs(dx), abs(dy))
        - hip_height: (floor_y - hip_mid_y) / ref_torso_len, clipped at -0.5
        - hip_speed: causal derivative of hip_height over 0.2s, clipped to [-5, 5]
        - aspect_ratio: bounding box width / height of visible landmarks
        - is_valid: boolean indicating whether torso core measurements are valid
    """
    n_frames = len(df)
    times = df["time_s"].to_numpy()

    # Landmark coordinates are normalized [0, 1] -> convert to pixel coordinates
    s_l_x = df[f"lm{LEFT_SHOULDER}_x"].to_numpy() * IMAGE_W
    s_l_y = df[f"lm{LEFT_SHOULDER}_y"].to_numpy() * IMAGE_H
    s_l_v = df[f"lm{LEFT_SHOULDER}_v"].to_numpy()

    s_r_x = df[f"lm{RIGHT_SHOULDER}_x"].to_numpy() * IMAGE_W
    s_r_y = df[f"lm{RIGHT_SHOULDER}_y"].to_numpy() * IMAGE_H
    s_r_v = df[f"lm{RIGHT_SHOULDER}_v"].to_numpy()

    h_l_x = df[f"lm{LEFT_HIP}_x"].to_numpy() * IMAGE_W
    h_l_y = df[f"lm{LEFT_HIP}_y"].to_numpy() * IMAGE_H
    h_l_v = df[f"lm{LEFT_HIP}_v"].to_numpy()

    h_r_x = df[f"lm{RIGHT_HIP}_x"].to_numpy() * IMAGE_W
    h_r_y = df[f"lm{RIGHT_HIP}_y"].to_numpy() * IMAGE_H
    h_r_v = df[f"lm{RIGHT_HIP}_v"].to_numpy()

    a_l_y = df[f"lm{LEFT_ANKLE}_y"].to_numpy() * IMAGE_H
    a_l_v = df[f"lm{LEFT_ANKLE}_v"].to_numpy()
    a_r_y = df[f"lm{RIGHT_ANKLE}_y"].to_numpy() * IMAGE_H
    a_r_v = df[f"lm{RIGHT_ANKLE}_v"].to_numpy()

    # Pre-allocate output arrays
    raw_torso_len_arr = np.full(n_frames, np.nan, dtype=np.float64)
    ref_torso_len_arr = np.full(n_frames, np.nan, dtype=np.float64)
    torso_angle_arr = np.full(n_frames, np.nan, dtype=np.float64)
    hip_height_arr = np.full(n_frames, np.nan, dtype=np.float64)
    hip_speed_arr = np.full(n_frames, np.nan, dtype=np.float64)
    aspect_ratio_arr = np.full(n_frames, np.nan, dtype=np.float64)
    is_valid_arr = np.zeros(n_frames, dtype=bool)

    h_mid_x_arr = np.full(n_frames, np.nan, dtype=np.float64)
    h_mid_y_arr = np.full(n_frames, np.nan, dtype=np.float64)

    # All landmarks matrix for aspect ratio: shape (n_frames, 33)
    all_x = np.column_stack([df[f"lm{i}_x"].to_numpy() * IMAGE_W for i in range(33)])
    all_y = np.column_stack([df[f"lm{i}_y"].to_numpy() * IMAGE_H for i in range(33)])
    all_v = np.column_stack([df[f"lm{i}_v"].to_numpy() for i in range(33)])

    prev_valid_idx: Optional[int] = None
    prev_h_x: Optional[float] = None
    prev_h_y: Optional[float] = None

    # 1. First Pass: Compute candidate torso measurements, reference length, and glitch removal
    for i in range(n_frames):
        # Check shoulder and hip visibility
        valid_torso = (
            (s_l_v[i] >= MIN_VISIBILITY)
            and (s_r_v[i] >= MIN_VISIBILITY)
            and (h_l_v[i] >= MIN_VISIBILITY)
            and (h_r_v[i] >= MIN_VISIBILITY)
            and not (
                np.isnan(s_l_x[i])
                or np.isnan(s_r_x[i])
                or np.isnan(h_l_x[i])
                or np.isnan(h_r_x[i])
            )
        )

        if not valid_torso:
            continue

        s_mid_x = (s_l_x[i] + s_r_x[i]) / 2.0
        s_mid_y = (s_l_y[i] + s_r_y[i]) / 2.0
        h_mid_x = (h_l_x[i] + h_r_x[i]) / 2.0
        h_mid_y = (h_l_y[i] + h_r_y[i]) / 2.0

        dx = h_mid_x - s_mid_x
        dy = h_mid_y - s_mid_y
        t_len = math.sqrt(dx * dx + dy * dy)

        if t_len < 1.0:
            continue

        raw_torso_len_arr[i] = t_len

        # Fix 1: Fold torso_angle into [0, 90] degrees using abs(dx) and abs(dy) with atan2
        # Flipped or unflipped gives identical angle; strictly capped at 90.0
        angle_rad = math.atan2(abs(dx), abs(dy))
        angle_deg = min(90.0, max(0.0, math.degrees(angle_rad)))
        torso_angle_arr[i] = angle_deg

        # Fix 2: Causal rolling median of torso_len over previous 3.0 seconds
        t_curr = times[i]
        t_start = t_curr - FLOOR_LOOKBACK_S  # 3.0s lookback
        past_mask = (times <= t_curr) & (times >= t_start) & (~np.isnan(raw_torso_len_arr))
        past_lens = raw_torso_len_arr[past_mask]

        ref_torso_len = float(np.median(past_lens)) if len(past_lens) > 0 else t_len

        # Torso length sanity check: mark invalid if current length is outside [0.5, 1.5] * ref
        if (t_len < 0.5 * ref_torso_len) or (t_len > 1.5 * ref_torso_len):
            raw_torso_len_arr[i] = np.nan
            torso_angle_arr[i] = np.nan
            continue

        # Fix 3: Remove tracking glitches
        # If hip midpoint moves more than 0.5 torso lengths in a single frame, mark invalid
        if (prev_valid_idx is not None) and (i == prev_valid_idx + 1):
            d_hip = math.sqrt((h_mid_x - prev_h_x) ** 2 + (h_mid_y - prev_h_y) ** 2)
            if d_hip > 0.5 * ref_torso_len:
                raw_torso_len_arr[i] = np.nan
                torso_angle_arr[i] = np.nan
                continue

        # Valid frame passed all sanity & glitch filters
        ref_torso_len_arr[i] = ref_torso_len
        h_mid_x_arr[i] = h_mid_x
        h_mid_y_arr[i] = h_mid_y
        is_valid_arr[i] = True

        prev_valid_idx = i
        prev_h_x = h_mid_x
        prev_h_y = h_mid_y

        # Aspect ratio of all visible landmarks (pixel units)
        vis_mask = (all_v[i] >= MIN_VISIBILITY) & (~np.isnan(all_x[i])) & (~np.isnan(all_y[i]))
        if np.count_nonzero(vis_mask) >= 2:
            pts_x = all_x[i, vis_mask]
            pts_y = all_y[i, vis_mask]
            bb_w = np.max(pts_x) - np.min(pts_x)
            bb_h = np.max(pts_y) - np.min(pts_y)
            if bb_h >= 1.0:
                aspect_ratio_arr[i] = bb_w / bb_h

    # 2. Compute Causal Rolling Floor Level and Hip Height
    # Floor is maximum ankle y over past FLOOR_LOOKBACK_S (image y points down -> largest y = lowest floor)
    ankle_y_per_frame = np.full(n_frames, np.nan, dtype=np.float64)
    for i in range(n_frames):
        ankles = []
        if (a_l_v[i] >= MIN_VISIBILITY) and not np.isnan(a_l_y[i]):
            ankles.append(a_l_y[i])
        if (a_r_v[i] >= MIN_VISIBILITY) and not np.isnan(a_r_y[i]):
            ankles.append(a_r_y[i])
        if ankles:
            ankle_y_per_frame[i] = max(ankles)

    # Causal sliding max for floor_y over FLOOR_LOOKBACK_S
    for i in range(n_frames):
        if not is_valid_arr[i]:
            continue

        t_curr = times[i]
        t_start = t_curr - FLOOR_LOOKBACK_S

        # Find frames in causal window [t_start, t_curr]
        window_mask = (times <= t_curr) & (times >= t_start)
        window_ankles = ankle_y_per_frame[window_mask]
        valid_ankles = window_ankles[~np.isnan(window_ankles)]

        if len(valid_ankles) > 0:
            floor_y = np.max(valid_ankles)
            h_y = h_mid_y_arr[i]
            # Fix 2: Normalize by reference torso length (not current frame's torso_len)
            ref_len = ref_torso_len_arr[i]
            height_val = (floor_y - h_y) / ref_len

            # Fix 5: Allow small negative values down to -0.5 (floor estimate noise), clip below -0.5
            hip_height_arr[i] = max(-0.5, height_val)

    # 3. Compute Causal Hip Speed (Derivative over ~0.2s lookback)
    for i in range(n_frames):
        if np.isnan(hip_height_arr[i]):
            continue

        t_curr = times[i]
        target_t = t_curr - SPEED_LOOKBACK_S

        best_j = -1
        best_diff = float("inf")

        for j in range(i - 1, -1, -1):
            if np.isnan(hip_height_arr[j]):
                continue
            dt = t_curr - times[j]
            if dt < 0.05:
                continue
            diff = abs(times[j] - target_t)
            if diff < best_diff:
                best_diff = diff
                best_j = j
            if times[j] < target_t - 0.15:
                break

        if best_j != -1:
            dt = t_curr - times[best_j]
            if dt >= 0.05:
                speed = (hip_height_arr[i] - hip_height_arr[best_j]) / dt
                # Fix 4: Clip hip_speed to [-5, 5] torso lengths per second
                hip_speed_arr[i] = max(-5.0, min(5.0, speed))

    # Ensure invalid frames have strictly NaN measurements
    for i in range(n_frames):
        if not is_valid_arr[i]:
            ref_torso_len_arr[i] = np.nan
            torso_angle_arr[i] = np.nan
            hip_height_arr[i] = np.nan
            hip_speed_arr[i] = np.nan
            aspect_ratio_arr[i] = np.nan

    result_df = pd.DataFrame(
        {
            "time_s": times,
            "torso_len": ref_torso_len_arr,
            "torso_angle": torso_angle_arr,
            "hip_height": hip_height_arr,
            "hip_speed": hip_speed_arr,
            "aspect_ratio": aspect_ratio_arr,
            "is_valid": is_valid_arr,
        }
    )

    return result_df


# ============================================================================
# 2. Window Features Extraction (Causal & Live-Compatible)
# ============================================================================

def compute_single_window_features(window_df: pd.DataFrame) -> Tuple[Dict[str, float], float]:
    """Calculate summary statistics (min, max, mean, std, change) across one window.

    Args:
        window_df: Subset DataFrame of measurements for the window.

    Returns:
        features_dict: Mapping of feature column name to scalar value.
        valid_fraction: Share of frames with valid measurements in the window.
    """
    total_frames = len(window_df)
    if total_frames == 0:
        return {col: np.nan for col in FEATURE_COLS}, 0.0

    valid_mask = window_df["is_valid"].to_numpy()
    valid_count = int(np.count_nonzero(valid_mask))
    valid_fraction = valid_count / total_frames

    feats: Dict[str, float] = {}

    for col in MEASUREMENT_COLS:
        vals = window_df[col].to_numpy()
        valid_vals = vals[~np.isnan(vals)]

        if len(valid_vals) > 0:
            feats[f"{col}_min"] = float(np.min(valid_vals))
            feats[f"{col}_max"] = float(np.max(valid_vals))
            feats[f"{col}_mean"] = float(np.mean(valid_vals))
            feats[f"{col}_std"] = float(np.std(valid_vals)) if len(valid_vals) > 1 else 0.0
            # Change: last valid minus first valid
            feats[f"{col}_change"] = float(valid_vals[-1] - valid_vals[0])
        else:
            feats[f"{col}_min"] = np.nan
            feats[f"{col}_max"] = np.nan
            feats[f"{col}_mean"] = np.nan
            feats[f"{col}_std"] = np.nan
            feats[f"{col}_change"] = np.nan

    return feats, valid_fraction


def window_features(
    measurements: pd.DataFrame,
    window_s: float = WINDOW_S,
    stride_s: float = STRIDE_S,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Slide a window over time and calculate statistical features.

    Args:
        measurements: DataFrame output from frame_measurements().
        window_s: Temporal window size in seconds (default 2.0s).
        stride_s: Temporal step size in seconds (default 0.25s).

    Returns:
        features_df: DataFrame with columns window_start, window_end, valid_fraction,
                     and all 20 statistical feature columns.
        feature_groups: Dict mapping group name ('angle', 'hip_height', 'speed', 'aspect')
                        to column names for ablation analysis.
    """
    if len(measurements) == 0:
        empty_cols = ["window_start", "window_end", "valid_fraction"] + FEATURE_COLS
        return pd.DataFrame(columns=empty_cols), FEATURE_GROUPS

    times = measurements["time_s"].to_numpy()
    t_min = float(times[0])
    t_max = float(times[-1])

    rows: List[Dict[str, Any]] = []

    w_start = t_min
    while w_start + window_s <= t_max + 1e-5:
        w_end = w_start + window_s

        w_mask = (times >= w_start - 1e-5) & (times <= w_end + 1e-5)
        w_df = measurements[w_mask]

        feats, valid_frac = compute_single_window_features(w_df)

        if valid_frac >= MIN_VALID_FRACTION:
            row = {
                "window_start": round(w_start, 4),
                "window_end": round(w_end, 4),
                "valid_fraction": round(valid_frac, 4),
            }
            row.update(feats)
            rows.append(row)

        w_start += stride_s

    if not rows:
        empty_cols = ["window_start", "window_end", "valid_fraction"] + FEATURE_COLS
        return pd.DataFrame(columns=empty_cols), FEATURE_GROUPS

    out_df = pd.DataFrame(rows)
    return out_df, FEATURE_GROUPS


# ============================================================================
# 3. Live Rolling Buffer Feature Extraction
# ============================================================================

def live_window_features(buffer: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Optional[Dict[str, float]]:
    """Extract features from the most recent window in a live rolling buffer.

    Guarantees exact code path parity with offline window_features().

    Args:
        buffer: Rolling buffer of frame measurements (DataFrame or list of measurement dicts).

    Returns:
        dict containing the 20 feature columns + valid_fraction, or None if valid_fraction < MIN_VALID_FRACTION.
    """
    if isinstance(buffer, list):
        if not buffer:
            return None
        buffer_df = pd.DataFrame(buffer)
    else:
        buffer_df = buffer

    if len(buffer_df) == 0:
        return None

    t_latest = float(buffer_df["time_s"].iloc[-1])
    t_start = t_latest - WINDOW_S
    window_df = buffer_df[buffer_df["time_s"] >= t_start]

    feats, valid_frac = compute_single_window_features(window_df)

    if valid_frac < MIN_VALID_FRACTION:
        return None

    result = {
        "window_start": round(t_start, 4),
        "window_end": round(t_latest, 4),
        "valid_fraction": round(valid_frac, 4),
    }
    result.update(feats)
    return result
