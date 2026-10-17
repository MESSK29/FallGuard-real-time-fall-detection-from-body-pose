"""build_windows.py - Sliding Window Dataset Generator and Labeling Pipeline.

Purpose:
--------
Iterates through all video landmark trajectories in `data/landmarks/*.parquet`,
computes causal kinematic measurements via `src.features`, slices 2.0-second sliding
temporal windows (0.25s stride), and applies the ground-truth fall coverage labeling rule.

Labeling Rule:
--------------
- For Fall videos: Window is labeled Fall (1) if it covers at least 50% (FALL_COVERAGE)
  of the annotated falling phase [fall_start_time, fall_end_time]; otherwise 0.
- For ADL videos: All windows are strictly labeled 0 (Non-Fall).

Output:
-------
`data/features/windows.parquet`
Columns: video_id, dataset, window_start, window_end, [20 feature columns], valid_fraction, label
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from src.features import (
    FALL_COVERAGE,
    FEATURE_COLS,
    MEASUREMENT_COLS,
    MIN_VALID_FRACTION,
    STRIDE_S,
    WINDOW_S,
    compute_single_window_features,
    frame_measurements,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fallguard.build_windows")


def process_video_windows(
    row: Dict[str, Any],
    landmarks_dir: Path,
) -> Tuple[List[Dict[str, Any]], int, int, Optional[pd.DataFrame]]:
    """Process a single video: load landmarks, extract features, and apply window labeling.

    Returns:
        valid_windows: List of window feature dictionary records.
        skipped_count: Number of candidate windows rejected due to low valid_fraction.
        fall_window_count: Number of positive fall windows produced for this video.
        meas_df: DataFrame of valid frame measurements for dataset-wide distribution analysis.
    """
    video_id = str(row["video_id"])
    dataset = str(row.get("dataset", "urfd"))
    label_str = str(row.get("label", "adl")).lower()
    fps = float(row.get("fps", 30.0))

    parquet_path = landmarks_dir / f"{video_id}.parquet"
    if not parquet_path.exists():
        logger.warning(f"Landmark parquet not found for {video_id}: {parquet_path}")
        return [], 0, 0, None

    df_lm = pd.read_parquet(parquet_path)
    if len(df_lm) == 0:
        return [], 0, 0, None

    # 1. Compute causal per-frame measurements
    measurements = frame_measurements(df_lm)
    times = measurements["time_s"].to_numpy()
    t_min = float(times[0])
    t_max = float(times[-1])

    # 2. Extract falling phase bounds if video is a fall
    is_fall_video = "fall" in label_str
    fall_start_time: Optional[float] = None
    fall_end_time: Optional[float] = None
    fall_duration: float = 0.0

    if is_fall_video:
        f_start = row.get("fall_start_frame")
        f_end = row.get("fall_end_frame")
        if pd.notna(f_start) and pd.notna(f_end):
            # URFD frame indices are 1-based in annotation metadata
            fall_start_time = max(0.0, (float(f_start) - 1.0) / fps)
            fall_end_time = max(fall_start_time, (float(f_end) - 1.0) / fps)
            fall_duration = fall_end_time - fall_start_time

    valid_windows: List[Dict[str, Any]] = []
    skipped_count = 0
    fall_window_count = 0

    # 3. Slide window over video duration
    w_start = t_min
    while w_start + WINDOW_S <= t_max + 1e-5:
        w_end = w_start + WINDOW_S

        w_mask = (times >= w_start - 1e-5) & (times <= w_end + 1e-5)
        w_df = measurements[w_mask]

        feats, valid_frac = compute_single_window_features(w_df)

        if valid_frac < MIN_VALID_FRACTION:
            skipped_count += 1
            w_start += STRIDE_S
            continue

        # 4. Labeling Logic
        window_label = 0
        if is_fall_video and (fall_start_time is not None) and (fall_duration > 0):
            # Overlap between window [w_start, w_end] and [fall_start_time, fall_end_time]
            overlap_start = max(w_start, fall_start_time)
            overlap_end = min(w_end, fall_end_time)
            overlap_sec = max(0.0, overlap_end - overlap_start)

            # Fall coverage: fraction of the falling phase covered by this window
            coverage = overlap_sec / fall_duration
            if coverage >= FALL_COVERAGE:
                window_label = 1
                fall_window_count += 1

        record = {
            "video_id": video_id,
            "dataset": dataset,
            "window_start": round(w_start, 4),
            "window_end": round(w_end, 4),
            "valid_fraction": round(valid_frac, 4),
            "label": int(window_label),
        }
        record.update(feats)
        valid_windows.append(record)

        w_start += STRIDE_S

    # Return valid frame measurements for distribution analysis
    valid_meas = measurements[measurements["is_valid"]][MEASUREMENT_COLS].copy()

    return valid_windows, skipped_count, fall_window_count, valid_meas


def build_windows(
    catalog_path: Path = Path("data/videos.csv"),
    landmarks_dir: Path = Path("data/landmarks"),
    output_parquet: Path = Path("data/features/windows.parquet"),
) -> pd.DataFrame:
    """Build standardized sliding window dataset across all indexed videos."""
    if not catalog_path.exists():
        raise FileNotFoundError(f"Master video catalog not found at: {catalog_path}")

    videos_df = pd.read_csv(catalog_path)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    all_windows: List[Dict[str, Any]] = []
    all_measurements: List[pd.DataFrame] = []
    total_skipped = 0
    zero_fall_videos: List[str] = []

    logger.info(f"Generating sliding windows for {len(videos_df)} video(s)...")

    for _, row in tqdm(videos_df.iterrows(), total=len(videos_df), desc="Building Windows"):
        video_id = str(row["video_id"])
        is_fall = "fall" in str(row.get("label", "")).lower()

        w_records, skipped, fall_cnt, valid_meas = process_video_windows(row.to_dict(), landmarks_dir)
        all_windows.extend(w_records)
        total_skipped += skipped
        if valid_meas is not None and len(valid_meas) > 0:
            all_measurements.append(valid_meas)

        if is_fall and fall_cnt == 0:
            zero_fall_videos.append(video_id)

    if not all_windows:
        logger.error("No windows could be generated! Check landmarks directory.")
        return pd.DataFrame()

    df_windows = pd.DataFrame(all_windows)

    # Ensure strictly specified column ordering:
    # video_id, dataset, window_start, window_end, [features], valid_fraction, label
    ordered_cols = (
        ["video_id", "dataset", "window_start", "window_end"]
        + FEATURE_COLS
        + ["valid_fraction", "label"]
    )
    df_windows = df_windows[ordered_cols]

    # Save to Parquet
    df_windows.to_parquet(output_parquet, index=False)
    logger.info(f"Saved window dataset to: {output_parquet} ({len(df_windows)} rows)")

    # ---------------------------------------------------------------------------
    # Statistical Diagnostics Report
    # ---------------------------------------------------------------------------
    total_windows = len(df_windows)
    fall_windows = int((df_windows["label"] == 1).sum())
    non_fall_windows = int((df_windows["label"] == 0).sum())

    print("\n" + "=" * 65)
    print("           FALLGUARD SLIDING WINDOW DATASET SUMMARY")
    print("=" * 65)
    print(f"Total Videos Processed     : {len(videos_df)}")
    print(f"Total Windows Generated    : {total_windows:,}")
    print(f"  - Fall Windows (Label=1) : {fall_windows:,} ({fall_windows / total_windows * 100:.2f}%)")
    print(f"  - Non-Fall (Label=0)     : {non_fall_windows:,} ({non_fall_windows / total_windows * 100:.2f}%)")
    print(f"Windows Skipped (Low Valid): {total_skipped:,}")
    print("-" * 65)

    if zero_fall_videos:
        print(f"[WARNING] The following {len(zero_fall_videos)} fall video(s) produced ZERO fall windows:")
        for vid in zero_fall_videos:
            print(f"   - {vid}")
    else:
        print("[OK] Every fall video successfully generated at least one positive fall window.")

    # Requirement 6: Print Min, Median, and Max of each of the 4 measurements
    if all_measurements:
        combined_meas = pd.concat(all_measurements, ignore_index=True)
        print("\n" + "-" * 65)
        print("DATASET MEASUREMENT DISTRIBUTIONS (Checking for Outliers):")
        print("-" * 65)
        for col in MEASUREMENT_COLS:
            vals = combined_meas[col].dropna()
            if len(vals) > 0:
                c_min = float(vals.min())
                c_med = float(vals.median())
                c_max = float(vals.max())
                print(f"  {col:<14} -> Min: {c_min:>8.3f} | Median: {c_med:>8.3f} | Max: {c_max:>8.3f}")
            else:
                print(f"  {col:<14} -> No valid values found")
        print("-" * 65)

    print("=" * 65 + "\n")
    return df_windows


if __name__ == "__main__":
    build_windows()
