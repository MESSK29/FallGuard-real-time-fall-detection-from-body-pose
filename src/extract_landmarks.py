"""extract_landmarks.py - MediaPipe Pose Landmark Extraction and Post-Processing.

Purpose:
--------
Reads video sequences indexed in `data/videos.csv` (supporting both image folders
and video files), runs MediaPipe Pose estimation on CPU, applies gap interpolation
and light smoothing, and serializes each video's landmark trajectory to `data/landmarks/{video_id}.parquet`.

Privacy Guarantee:
------------------
Only extracted 33-point landmark coordinates (x, y, z, visibility) are written to disk.
Zero raw images, frames, or video clips are saved or cached in the output.
"""

import argparse
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fallguard.extract_landmarks")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

NUM_LANDMARKS = 33
MAX_INTERPOLATION_GAP = 4
SMOOTHING_WINDOW = 3


def natural_sort_key(file_path: Path) -> int:
    """Extract numeric suffix from filename for natural sequential sorting."""
    numbers = re.findall(r"\d+", file_path.stem)
    if numbers:
        return int(numbers[-1])
    return 0


def get_image_folder_frames(folder_path: Path) -> List[Path]:
    """Retrieve all image files in folder sorted by their internal frame number."""
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files: List[Path] = []
    for ext in extensions:
        files.extend(folder_path.glob(ext))
    files.sort(key=natural_sort_key)
    return files


def load_frame_stream(path: Path) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Yield sequential (frame_index, bgr_image) from either image folder or video file."""
    if path.is_dir():
        image_files = get_image_folder_frames(path)
        for idx, img_path in enumerate(image_files):
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to read image file: {img_path}")
                continue
            yield idx, img
    elif path.is_file():
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {path}")
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            yield idx, frame
            idx += 1
        cap.release()
    else:
        raise FileNotFoundError(f"Input path does not exist: {path}")


def interpolate_and_smooth(df: pd.DataFrame) -> pd.DataFrame:
    """Post-process landmark trajectories per video.

    1. Linear interpolation for gaps <= 4 consecutive missing frames.
       Does NOT fill gaps > 4 frames. Does NOT extrapolate at sequence start or end.
    2. Centered moving average over 3 frames on (x, y, z) coordinates only (not visibility).
    3. Preserves original 'detected' column and marks filled frames in 'interpolated'.
    """
    df = df.copy()
    n = len(df)
    df["interpolated"] = False

    if n == 0:
        return df

    # Landmark coordinate column names
    coord_cols = []
    for i in range(NUM_LANDMARKS):
        coord_cols.extend([f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"])

    all_lm_cols = []
    for i in range(NUM_LANDMARKS):
        all_lm_cols.extend([f"lm{i}_x", f"lm{i}_y", f"lm{i}_z", f"lm{i}_v"])

    # 1. Bounded Gap Interpolation
    i = 0
    while i < n:
        if not df.loc[i, "detected"]:
            start = i
            while i < n and not df.loc[i, "detected"]:
                i += 1
            end = i - 1
            gap_len = end - start + 1

            # Only interpolate if interior (not at start, not at end) and gap <= 4 frames
            if start > 0 and end < n - 1 and gap_len <= MAX_INTERPOLATION_GAP:
                for col in all_lm_cols:
                    val_before = df.loc[start - 1, col]
                    val_after = df.loc[end + 1, col]
                    if pd.notna(val_before) and pd.notna(val_after):
                        for k in range(gap_len):
                            alpha = (k + 1) / (gap_len + 1)
                            interp_val = (1.0 - alpha) * val_before + alpha * val_after
                            df.loc[start + k, col] = interp_val

                df.loc[start:end, "interpolated"] = True
        else:
            i += 1

    # 2. Centered Moving Average (3 frames) on (x, y, z) only
    valid_mask = df["detected"] | df["interpolated"]
    smoothed = df[coord_cols].rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1).mean()

    # Re-apply NaN to any frames that were not detected or interpolated
    smoothed[~valid_mask] = np.nan
    df[coord_cols] = smoothed

    return df


def calculate_longest_gap(detected_series: pd.Series) -> int:
    """Calculate the longest run of consecutive missing frames in raw detections."""
    max_gap = 0
    current_gap = 0
    for det in detected_series:
        if not det:
            current_gap += 1
            if current_gap > max_gap:
                max_gap = current_gap
        else:
            current_gap = 0
    return max_gap


def extract_video_landmarks(
    video_info: Dict[str, Any],
    output_parquet: Path,
    is_debug: bool = False,
    debug_video_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run MediaPipe Pose on a single video sequence, process, and persist Parquet."""
    video_id = str(video_info["video_id"])
    path = Path(video_info["path"])
    expected_frames = int(video_info.get("n_frames", 0))
    fps = float(video_info.get("fps", 30.0))
    label = str(video_info.get("label", "unknown"))

    rows: List[Dict[str, Any]] = []

    # Optional debug video writer
    debug_writer: Optional[cv2.VideoWriter] = None

    # Requirement 2: Dedicated Pose instance per video so state does not leak
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        for frame_idx, bgr_frame in load_frame_stream(path):
            h, w = bgr_frame.shape[:2]

            if is_debug and debug_video_path and debug_writer is None:
                debug_video_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                debug_writer = cv2.VideoWriter(str(debug_video_path), fourcc, fps, (w, h))

            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            detected = False
            frame_data: Dict[str, Any] = {
                "video_id": video_id,
                "frame": frame_idx,
                "time_s": round(frame_idx / fps, 4),
            }

            if results.pose_landmarks and len(results.pose_landmarks.landmark) == NUM_LANDMARKS:
                detected = True
                frame_data["detected"] = True
                for idx, lm in enumerate(results.pose_landmarks.landmark):
                    frame_data[f"lm{idx}_x"] = float(lm.x)
                    frame_data[f"lm{idx}_y"] = float(lm.y)
                    frame_data[f"lm{idx}_z"] = float(lm.z)
                    frame_data[f"lm{idx}_v"] = float(lm.visibility) if lm.visibility is not None else 1.0

                if is_debug and debug_writer is not None:
                    mp_drawing.draw_landmarks(
                        bgr_frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )
                    cv2.putText(
                        bgr_frame,
                        f"Frame: {frame_idx} | DETECTED",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
            else:
                frame_data["detected"] = False
                for idx in range(NUM_LANDMARKS):
                    frame_data[f"lm{idx}_x"] = np.nan
                    frame_data[f"lm{idx}_y"] = np.nan
                    frame_data[f"lm{idx}_z"] = np.nan
                    frame_data[f"lm{idx}_v"] = np.nan

                if is_debug and debug_writer is not None:
                    cv2.putText(
                        bgr_frame,
                        f"Frame: {frame_idx} | NO DETECTION",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        bgr_frame,
                        "NO DETECTION",
                        (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3,
                        cv2.LINE_AA,
                    )

            if is_debug and debug_writer is not None:
                debug_writer.write(bgr_frame)

            rows.append(frame_data)

    if debug_writer is not None:
        debug_writer.release()
        logger.info(f"Saved debug preview video: {debug_video_path}")

    actual_frames = len(rows)
    if expected_frames > 0 and actual_frames != expected_frames:
        logger.warning(
            f"Frame count mismatch for {video_id}: processed {actual_frames} frames, "
            f"expected {expected_frames} from metadata."
        )

    if actual_frames == 0:
        logger.error(f"Zero frames extracted for video: {video_id}")
        return {
            "video_id": video_id,
            "label": label,
            "n_frames": 0,
            "detection_rate": 0.0,
            "longest_missing_gap": 0,
        }

    raw_df = pd.DataFrame(rows)

    # Compute raw detection metrics before interpolation
    detected_count = int(raw_df["detected"].sum())
    detection_rate = round((detected_count / actual_frames) * 100.0, 2)
    longest_gap = calculate_longest_gap(raw_df["detected"])

    # Step 4: Interpolate gaps <= 4 and apply 3-frame centered moving average
    processed_df = interpolate_and_smooth(raw_df)

    # Step 5: Save to Parquet
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_parquet(output_parquet, index=False)

    logger.info(
        f"[{video_id}] {actual_frames} frames | Detection rate: {detection_rate}% | "
        f"Longest gap: {longest_gap} frames | Saved: {output_parquet.name}"
    )

    return {
        "video_id": video_id,
        "label": label,
        "n_frames": actual_frames,
        "detection_rate": detection_rate,
        "longest_missing_gap": longest_gap,
    }


def select_debug_videos(videos_df: pd.DataFrame, seed: int = 42) -> Set[str]:
    """Deterministically pick 2 random fall videos and 1 random ADL video for debug rendering."""
    rng = random.Random(seed)
    fall_ids = videos_df[videos_df["label"] == "fall"]["video_id"].tolist()
    adl_ids = videos_df[videos_df["label"] == "adl"]["video_id"].tolist()

    selected_falls = rng.sample(fall_ids, min(2, len(fall_ids))) if fall_ids else []
    selected_adls = rng.sample(adl_ids, min(1, len(adl_ids))) if adl_ids else []

    selected_set = set(selected_falls + selected_adls)
    logger.info(f"Selected debug preview videos (seed={seed}): {selected_set}")
    return selected_set


def run_pipeline(
    catalog_path: Path = Path("data/videos.csv"),
    output_dir: Path = Path("data/landmarks"),
    reports_dir: Path = Path("reports"),
    force: bool = False,
    debug: bool = False,
    limit: Optional[int] = None,
) -> None:
    """Execute landmark extraction across all videos with progress reporting."""
    if not catalog_path.exists():
        raise FileNotFoundError(f"Master video catalog not found at: {catalog_path}. Run prepare_datasets.py first.")

    videos_df = pd.read_csv(catalog_path)
    if limit is not None and limit > 0:
        logger.info(f"Limiting execution to first {limit} video(s).")
        videos_df = videos_df.head(limit)

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    debug_ids: Set[str] = set()
    if debug:
        debug_ids = select_debug_videos(videos_df)

    summary_records: List[Dict[str, Any]] = []

    logger.info(f"Beginning landmark extraction for {len(videos_df)} video sequence(s)...")

    # Step 6: tqdm progress bar over videos
    for _, row in tqdm(videos_df.iterrows(), total=len(videos_df), desc="Extracting Pose Landmarks"):
        video_id = str(row["video_id"])
        parquet_path = output_dir / f"{video_id}.parquet"

        # Skip existing if not --force
        if parquet_path.exists() and not force:
            logger.info(f"Skipping {video_id}: already extracted ({parquet_path.name}). Use --force to redo.")
            try:
                existing_df = pd.read_parquet(parquet_path)
                det_rate = round((existing_df["detected"].sum() / len(existing_df)) * 100.0, 2)
                l_gap = calculate_longest_gap(existing_df["detected"])
                summary_records.append({
                    "video_id": video_id,
                    "label": row["label"],
                    "n_frames": len(existing_df),
                    "detection_rate": det_rate,
                    "longest_missing_gap": l_gap,
                })
            except Exception as e:
                logger.warning(f"Could not read existing parquet for {video_id}: {e}")
            continue

        is_debug_video = debug and (video_id in debug_ids)
        debug_video_file = reports_dir / "debug" / f"{video_id}_debug.mp4" if is_debug_video else None

        # Step 6: Wrap each video in try/except so one failure does not halt execution
        try:
            metrics = extract_video_landmarks(
                video_info=row.to_dict(),
                output_parquet=parquet_path,
                is_debug=is_debug_video,
                debug_video_path=debug_video_file,
            )
            summary_records.append(metrics)
        except Exception as exc:
            logger.error(f"Error processing video {video_id}: {exc}", exc_info=True)
            summary_records.append({
                "video_id": video_id,
                "label": row.get("label", "unknown"),
                "n_frames": 0,
                "detection_rate": 0.0,
                "longest_missing_gap": -1,
            })

    # Step 7: Save reports/extraction_summary.csv and print analytical summary
    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_csv = reports_dir / "extraction_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f"Saved extraction summary to: {summary_csv}")

        print("\n" + "=" * 65)
        print("          FALLGUARD LANDMARK EXTRACTION SUMMARY")
        print("=" * 65)

        # Average detection rate per class
        fall_df = summary_df[summary_df["label"] == "fall"]
        adl_df = summary_df[summary_df["label"] == "adl"]

        avg_fall_det = fall_df["detection_rate"].mean() if not fall_df.empty else 0.0
        avg_adl_det = adl_df["detection_rate"].mean() if not adl_df.empty else 0.0
        overall_avg = summary_df["detection_rate"].mean() if not summary_df.empty else 0.0

        print(f"Overall Processed : {len(summary_df)} videos ({summary_df['n_frames'].sum():,} frames)")
        print(f"Overall Detection : {overall_avg:.2f}%")
        print(f"Falls Detection   : {avg_fall_det:.2f}% (across {len(fall_df)} videos)")
        print(f"ADLs Detection    : {avg_adl_det:.2f}% (across {len(adl_df)} videos)")
        print("-" * 65)

        # 5 videos with lowest detection rate
        lowest_5 = summary_df.sort_values(by="detection_rate", ascending=True).head(5)
        print("5 Videos with Lowest Detection Rate:")
        for _, r in lowest_5.iterrows():
            print(
                f"  - {r['video_id']:<10} ({r['label']}): {r['detection_rate']:>6.2f}% detected | "
                f"Longest gap: {r['longest_missing_gap']} frames | Total: {r['n_frames']} frames"
            )
        print("=" * 65 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract MediaPipe 33-landmark pose trajectories.")
    parser.add_argument("--catalog", type=Path, default=Path("data/videos.csv"), help="Path to videos.csv catalog")
    parser.add_argument("--output", type=Path, default=Path("data/landmarks"), help="Output directory for Parquet files")
    parser.add_argument("--reports", type=Path, default=Path("reports"), help="Reports destination directory")
    parser.add_argument("--force", action="store_true", help="Force re-extraction of existing parquet files")
    parser.add_argument("--debug", action="store_true", help="Render debug preview videos for 2 falls and 1 ADL")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N videos for quick testing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        catalog_path=args.catalog,
        output_dir=args.output,
        reports_dir=args.reports,
        force=args.force,
        debug=args.debug,
        limit=args.limit,
    )
