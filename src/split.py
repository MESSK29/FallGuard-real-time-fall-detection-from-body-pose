"""split.py - Stratified Train/Test Dataset Split for Video Sequences.

Hold out 20% of VIDEOS as the final test set, stratified by label (fall/adl),
with random_state=42. Guarantees that at least one lying ADL (has_lying_adl=True)
is present in the test set.

Saves:
- models/train_videos.txt
- models/test_videos.txt

Exports:
- load_split(force=False) -> (train_ids, test_ids)
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_SPLIT_PATH = Path("models/train_videos.txt")
TEST_SPLIT_PATH = Path("models/test_videos.txt")
VIDEOS_CSV_PATH = Path("data/videos.csv")


def create_split(
    videos_csv: Union[str, Path] = VIDEOS_CSV_PATH,
    train_path: Union[str, Path] = TRAIN_SPLIT_PATH,
    test_path: Union[str, Path] = TEST_SPLIT_PATH,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[List[str], List[str]]:
    """Create a stratified video-level train/test split and persist to disk.

    Args:
        videos_csv: Path to data/videos.csv.
        train_path: Destination path for models/train_videos.txt.
        test_path: Destination path for models/test_videos.txt.
        test_size: Fraction of videos held out for testing (0.20).
        random_state: Seed for reproducibility (42).

    Returns:
        (train_ids, test_ids): Lists of video ID strings.
    """
    videos_csv = Path(videos_csv)
    train_path = Path(train_path)
    test_path = Path(test_path)

    if not videos_csv.exists():
        raise FileNotFoundError(f"Video catalog not found: {videos_csv}")

    df = pd.read_csv(videos_csv)

    # Stratified split by label (fall vs adl)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    # Ensure has_lying_adl is boolean
    if "has_lying_adl" in test_df.columns:
        test_has_lying = test_df["has_lying_adl"].astype(bool)
        lying_in_test = test_df[test_has_lying & (test_df["label"] == "adl")]

        # Requirement: Test set must contain at least 1 ADL with has_lying_adl = True
        if len(lying_in_test) == 0:
            train_has_lying = train_df["has_lying_adl"].astype(bool)
            train_lying_adls = train_df[train_has_lying & (train_df["label"] == "adl")].sort_values("video_id")

            if len(train_lying_adls) == 0:
                raise ValueError("No lying ADLs available in dataset to swap into test set.")

            # Choose the first lying ADL from train to swap into test
            swap_in = train_lying_adls.iloc[0]

            # Choose a non-lying ADL from test to swap into train
            test_non_lying_adls = test_df[(~test_has_lying) & (test_df["label"] == "adl")].sort_values("video_id")
            if len(test_non_lying_adls) == 0:
                # If all test ADLs were somehow marked True, fallback to any test ADL
                test_non_lying_adls = test_df[test_df["label"] == "adl"].sort_values("video_id")

            swap_out = test_non_lying_adls.iloc[-1]

            # Perform the swap
            train_df = train_df[train_df["video_id"] != swap_in["video_id"]]
            test_df = test_df[test_df["video_id"] != swap_out["video_id"]]

            train_df = pd.concat([train_df, pd.DataFrame([swap_out])], ignore_index=True)
            test_df = pd.concat([test_df, pd.DataFrame([swap_in])], ignore_index=True)

    # Sort video IDs for deterministic file output
    train_ids = sorted(train_df["video_id"].astype(str).tolist())
    test_ids = sorted(test_df["video_id"].astype(str).tolist())

    # Assert no overlap
    overlap = set(train_ids) & set(test_ids)
    assert len(overlap) == 0, f"Data leakage error: overlapping video IDs found: {overlap}"

    # Save to files
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    with open(train_path, "w", encoding="utf-8") as f:
        for vid in train_ids:
            f.write(f"{vid}\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for vid in test_ids:
            f.write(f"{vid}\n")

    return train_ids, test_ids


def load_split(
    videos_csv: Union[str, Path] = VIDEOS_CSV_PATH,
    train_path: Union[str, Path] = TRAIN_SPLIT_PATH,
    test_path: Union[str, Path] = TEST_SPLIT_PATH,
    force: bool = False,
) -> Tuple[List[str], List[str]]:
    """Load train and test video IDs. If files do not exist or force=True, re-splits.

    Args:
        videos_csv: Path to videos.csv.
        train_path: Path to models/train_videos.txt.
        test_path: Path to models/test_videos.txt.
        force: If True, redo the split even if split files exist.

    Returns:
        (train_ids, test_ids): Lists of video ID strings.
    """
    train_path = Path(train_path)
    test_path = Path(test_path)

    if train_path.exists() and test_path.exists() and not force:
        with open(train_path, "r", encoding="utf-8") as f:
            train_ids = [line.strip() for line in f if line.strip()]
        with open(test_path, "r", encoding="utf-8") as f:
            test_ids = [line.strip() for line in f if line.strip()]

        overlap = set(train_ids) & set(test_ids)
        assert len(overlap) == 0, f"Data leakage error: overlapping video IDs found: {overlap}"
        return train_ids, test_ids

    return create_split(videos_csv=videos_csv, train_path=train_path, test_path=test_path)


def print_split_summary(
    train_ids: List[str],
    test_ids: List[str],
    videos_csv: Union[str, Path] = VIDEOS_CSV_PATH,
) -> None:
    """Print the counts of falls/ADLs in train and test sets, and verify assertions."""
    overlap = set(train_ids) & set(test_ids)
    assert len(overlap) == 0, f"Overlap assertion failed: {overlap}"

    df = pd.read_csv(videos_csv)
    df_train = df[df["video_id"].isin(train_ids)]
    df_test = df[df["video_id"].isin(test_ids)]

    train_falls = int((df_train["label"] == "fall").sum())
    train_adls = int((df_train["label"] == "adl").sum())
    test_falls = int((df_test["label"] == "fall").sum())
    test_adls = int((df_test["label"] == "adl").sum())

    train_lying = int(df_train["has_lying_adl"].astype(bool).sum())
    test_lying = int(df_test["has_lying_adl"].astype(bool).sum())

    print("\n" + "=" * 60)
    print("         FALLGUARD STRATIFIED TRAIN / TEST SPLIT")
    print("=" * 60)
    print(f"Total Videos         : {len(df)}")
    print(f"Train Set ({len(train_ids)} videos) : {train_falls} Falls | {train_adls} ADLs ({train_lying} lying ADLs)")
    print(f"Test Set  ({len(test_ids)} videos) : {test_falls} Falls | {test_adls} ADLs ({test_lying} lying ADLs)")
    print(f"Overlap Check        : [OK] 0 overlapping videos")
    print(f"Train file           : {TRAIN_SPLIT_PATH}")
    print(f"Test file            : {TEST_SPLIT_PATH}")
    print("=" * 60)
    print("Test video IDs       : " + ", ".join(test_ids))
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or inspect video-level train/test split.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-creation of train and test split files even if they already exist.",
    )
    args = parser.parse_args()

    train_ids, test_ids = load_split(force=args.force)
    print_split_summary(train_ids, test_ids)


if __name__ == "__main__":
    main()
