"""prepare_datasets.py - Dataset Indexing and Standardization Pipeline.

Purpose:
--------
Scans raw fall and ADL (Activities of Daily Living) dataset directories (e.g. URFD
and optional custom clips) and generates a standardized master catalog: `data/videos.csv`.

Columns in data/videos.csv:
---------------------------
- video_id: Unique sequence identifier (e.g., 'fall-01', 'adl-01')
- dataset: Dataset name ('urfd', 'own')
- path: Relative path to sequence folder or video file
- label: 'fall' or 'adl'
- fps: Sampling frame rate (30.0 for URFD, or detected via OpenCV)
- n_frames: Total number of frames in the sequence
- fall_start_frame: First frame where fall begins (label == 0)
- fall_end_frame: First frame where lying begins (label == 1) after fall_start
- has_lying_adl: Boolean indicating if an ADL sequence includes lying postures

URFD Ground Truth Label Scheme:
-------------------------------
-1 : Not lying (standing, walking, sitting, bending)
 0 : Falling / transition phase
 1 : Lying on floor
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fallguard.prepare_datasets")


def load_urfd_csv(csv_path: Path) -> pd.DataFrame:
    """Load a URFD label CSV file, inspecting and handling header presence.

    Columns in URFD CSV:
    0: Sequence name (e.g., 'fall-01', 'adl-01')
    1: Frame number (integer)
    2: Posture label (-1: not lying, 0: falling, 1: lying)
    3..: Optional extracted depth/bounding features
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"URFD annotation CSV not found: {csv_path}")

    # Inspect the first line to verify if a header exists
    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    first_tokens = [tok.strip() for tok in first_line.split(",")]
    has_header = False
    # If the second token cannot be converted to int, it's a header line
    try:
        int(first_tokens[1])
    except (ValueError, IndexError):
        has_header = True

    if has_header:
        df = pd.read_csv(csv_path, header=0)
        # Rename standard columns if needed
        df.columns = ["sequence", "frame_num", "label"] + list(df.columns[3:])
    else:
        df = pd.read_csv(csv_path, header=None)
        col_names = ["sequence", "frame_num", "label"] + [f"feat_{i}" for i in range(len(df.columns) - 3)]
        df.columns = col_names

    # Clean and cast types
    df["sequence"] = df["sequence"].astype(str).str.strip()
    df["frame_num"] = pd.to_numeric(df["frame_num"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["frame_num", "label"])
    df["frame_num"] = df["frame_num"].astype(int)
    df["label"] = df["label"].astype(int)

    return df


def parse_urfd_fall_keyframes(seq_df: pd.DataFrame) -> Tuple[Optional[int], Optional[int]]:
    """Determine fall_start_frame and fall_end_frame from sequence label series.

    - fall_start_frame: First frame labeled 0 (falling transition).
    - fall_end_frame: First frame labeled 1 (lying down) occurring after fall_start_frame.
    """
    sorted_df = seq_df.sort_values(by="frame_num")
    zero_rows = sorted_df[sorted_df["label"] == 0]

    if zero_rows.empty:
        return None, None

    fall_start = int(zero_rows.iloc[0]["frame_num"])

    # Find first lying frame (1) after the fall starts
    lying_after_fall = sorted_df[(sorted_df["frame_num"] > fall_start) & (sorted_df["label"] == 1)]
    if lying_after_fall.empty:
        return fall_start, None

    fall_end = int(lying_after_fall.iloc[0]["frame_num"])
    return fall_start, fall_end


def scan_urfd(raw_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Scan URFD directories (falls/ and adls/) and correlate with annotation CSVs.

    Returns:
        records: List of video dictionary records.
        uncomputed_falls: List of sequence names where fall keyframes could not be computed.
    """
    urfd_dir = raw_dir / "urfd"
    if not urfd_dir.exists():
        logger.warning(f"URFD directory not found at: {urfd_dir}")
        return [], []

    falls_csv_path = urfd_dir / "urfall-cam0-falls.csv"
    adls_csv_path = urfd_dir / "urfall-cam0-adls.csv"

    falls_df = load_urfd_csv(falls_csv_path) if falls_csv_path.exists() else pd.DataFrame()
    adls_df = load_urfd_csv(adls_csv_path) if adls_csv_path.exists() else pd.DataFrame()

    records: List[Dict[str, Any]] = []
    uncomputed_falls: List[str] = []

    # 1. Process Fall Sequences
    falls_dir = urfd_dir / "falls"
    if falls_dir.exists():
        fall_folders = sorted([d for d in falls_dir.iterdir() if d.is_dir()])
        for folder in fall_folders:
            seq_name = folder.name.replace("-cam0-rgb", "").strip()
            png_files = list(folder.glob("*.png"))
            n_frames = len(png_files)

            seq_labels = falls_df[falls_df["sequence"] == seq_name] if not falls_df.empty else pd.DataFrame()

            if seq_labels.empty:
                logger.warning(f"[URFD] No matching label records found in CSV for fall folder: {folder.name}")
                fall_start, fall_end = None, None
            else:
                csv_frames = len(seq_labels)
                if csv_frames != n_frames:
                    logger.warning(
                        f"[URFD] Frame count mismatch for {seq_name}: "
                        f"{n_frames} PNGs on disk vs {csv_frames} rows in CSV."
                    )

                fall_start, fall_end = parse_urfd_fall_keyframes(seq_labels)
                if fall_start is None or fall_end is None:
                    uncomputed_falls.append(seq_name)

            records.append({
                "video_id": seq_name,
                "dataset": "urfd",
                "path": str(folder.as_posix()),
                "label": "fall",
                "fps": 30.0,
                "n_frames": n_frames,
                "fall_start_frame": fall_start if fall_start is not None else pd.NA,
                "fall_end_frame": fall_end if fall_end is not None else pd.NA,
                "has_lying_adl": False,
            })

    # 2. Process ADL Sequences (only existing folders)
    adls_dir = urfd_dir / "adls"
    if adls_dir.exists():
        adl_folders = sorted([d for d in adls_dir.iterdir() if d.is_dir()])
        for folder in adl_folders:
            seq_name = folder.name.replace("-cam0-rgb", "").strip()
            png_files = list(folder.glob("*.png"))
            n_frames = len(png_files)

            seq_labels = adls_df[adls_df["sequence"] == seq_name] if not adls_df.empty else pd.DataFrame()

            has_lying = False
            if seq_labels.empty:
                logger.warning(f"[URFD] No matching label records found in CSV for ADL folder: {folder.name}")
            else:
                csv_frames = len(seq_labels)
                if csv_frames != n_frames:
                    logger.warning(
                        f"[URFD] Frame count mismatch for {seq_name}: "
                        f"{n_frames} PNGs on disk vs {csv_frames} rows in CSV."
                    )

                # has_lying_adl = True if ADL sequence has any frame labeled 1
                has_lying = bool((seq_labels["label"] == 1).any())

            records.append({
                "video_id": seq_name,
                "dataset": "urfd",
                "path": str(folder.as_posix()),
                "label": "adl",
                "fps": 30.0,
                "n_frames": n_frames,
                "fall_start_frame": pd.NA,
                "fall_end_frame": pd.NA,
                "has_lying_adl": has_lying,
            })

    return records, uncomputed_falls


def scan_own_clips(own_csv_path: Path) -> List[Dict[str, Any]]:
    """Scan optional custom user video clips specified in data/own_clips.csv.

    Uses OpenCV to extract native fps and total frame count.
    Silently returns empty list if data/own_clips.csv is absent.
    """
    if not own_csv_path.exists():
        return []

    logger.info(f"Detected custom clips catalog at: {own_csv_path}")
    try:
        df_own = pd.read_csv(own_csv_path)
    except Exception as e:
        logger.error(f"Error reading {own_csv_path}: {e}")
        return []

    records: List[Dict[str, Any]] = []
    for _, row in df_own.iterrows():
        video_path_str = str(row.get("path", "")).strip()
        video_path = Path(video_path_str)

        fps = 30.0
        n_frames = 0

        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                detected_fps = cap.get(cv2.CAP_PROP_FPS)
                if detected_fps and detected_fps > 0:
                    fps = round(float(detected_fps), 2)
                detected_n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if detected_n and detected_n > 0:
                    n_frames = int(detected_n)
                cap.release()
            else:
                logger.warning(f"[Own Dataset] Could not open video file via OpenCV: {video_path}")
        else:
            logger.warning(f"[Own Dataset] Video path does not exist on disk: {video_path}")

        lbl = str(row.get("label", "adl")).lower().strip()
        label_val = "fall" if "fall" in lbl else "adl"

        fall_start = row.get("fall_start_frame", pd.NA)
        fall_end = row.get("fall_end_frame", pd.NA)
        has_lying = bool(row.get("has_lying_adl", False))

        video_id = str(row.get("video_id", video_path.stem)).strip()

        records.append({
            "video_id": video_id,
            "dataset": "own",
            "path": video_path.as_posix() if video_path.exists() else video_path_str,
            "label": label_val,
            "fps": fps,
            "n_frames": n_frames,
            "fall_start_frame": fall_start if pd.notna(fall_start) else pd.NA,
            "fall_end_frame": fall_end if pd.notna(fall_end) else pd.NA,
            "has_lying_adl": has_lying if label_val == "adl" else False,
        })

    return records


def prepare_datasets(
    data_dir: Path = Path("data"),
    output_csv: Path = Path("data/videos.csv"),
) -> pd.DataFrame:
    """Scan raw datasets, build data/videos.csv, and display comprehensive summary."""
    raw_dir = data_dir / "raw"
    own_csv_path = data_dir / "own_clips.csv"

    logger.info(f"Scanning raw dataset directories in: {raw_dir}")
    urfd_records, uncomputed_falls = scan_urfd(raw_dir)
    own_records = scan_own_clips(own_csv_path)

    all_records = urfd_records + own_records

    if not all_records:
        logger.error("No video sequences found across data/raw! data/videos.csv will be empty.")
        df_empty = pd.DataFrame(columns=[
            "video_id", "dataset", "path", "label", "fps", "n_frames",
            "fall_start_frame", "fall_end_frame", "has_lying_adl"
        ])
        df_empty.to_csv(output_csv, index=False)
        return df_empty

    df = pd.DataFrame(all_records)

    # Ensure strictly specified column ordering
    cols = [
        "video_id",
        "dataset",
        "path",
        "label",
        "fps",
        "n_frames",
        "fall_start_frame",
        "fall_end_frame",
        "has_lying_adl",
    ]
    df = df[cols]

    # Save to data/videos.csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved master dataset index to: {output_csv} ({len(df)} entries)")

    # ---------------------------------------------------------------------------
    # Summary & Diagnostics Report
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("           FALLGUARD DATASET PREPARATION SUMMARY")
    print("=" * 65)

    total_videos = len(df)
    total_frames = df["n_frames"].sum()

    for ds_name, ds_group in df.groupby("dataset"):
        n_fall = len(ds_group[ds_group["label"] == "fall"])
        n_adl = len(ds_group[ds_group["label"] == "adl"])
        ds_frames = ds_group["n_frames"].sum()
        lying_adls = len(ds_group[(ds_group["label"] == "adl") & (ds_group["has_lying_adl"] == True)])

        print(f"\nDataset: {ds_name.upper()}")
        print(f"  - Fall sequences : {n_fall}")
        print(f"  - ADL sequences  : {n_adl}")
        print(f"  - Total frames   : {ds_frames:,}")
        print(f"  - ADLs with lying: {lying_adls} of {n_adl}")

    overall_falls = len(df[df["label"] == "fall"])
    overall_adls = len(df[df["label"] == "adl"])
    overall_lying_adls = len(df[(df["label"] == "adl") & (df["has_lying_adl"] == True)])

    print("-" * 65)
    print(f"OVERALL TOTALS:")
    print(f"  - Total Videos   : {total_videos}")
    print(f"  - Falls / ADLs   : {overall_falls} falls / {overall_adls} ADLs")
    print(f"  - Total Frames   : {total_frames:,}")
    print(f"  - Lying ADLs     : {overall_lying_adls} (Crucial hard negatives for training)")
    print("=" * 65)

    # Dataset size warnings (< 15 check)
    if overall_falls < 15:
        print(f"\n[WARNING] Found only {overall_falls} fall videos! Project minimum recommended is 15.")
    if overall_adls < 15:
        print(f"\n[WARNING] Found only {overall_adls} ADL videos! Project minimum recommended is 15.")

    # Uncomputed fall keyframes check
    if uncomputed_falls:
        print("\n[WARNING] Fall start/end keyframes could NOT be computed for the following videos:")
        for v in uncomputed_falls:
            print(f"   - {v}")
    else:
        print("\n[OK] All fall videos have valid, computed fall_start_frame and fall_end_frame keyframes.")

    print("=" * 65 + "\n")
    return df


if __name__ == "__main__":
    prepare_datasets()
