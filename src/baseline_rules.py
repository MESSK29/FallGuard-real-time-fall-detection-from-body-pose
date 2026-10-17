"""baseline_rules.py - Heuristic Baseline Fall Detection Rule Model.

Applies kinematic threshold rule:
    Predict Fall (1) if:
        torso_angle_max > A
        AND hip_speed_min < -S   (downward vertical speed)
        AND hip_height_mean < H

Tunes thresholds (A, S, H) via grid search maximizing F1 score strictly on
the TRAINING split from src.split (never observing test data).

Outputs:
- models/baseline_thresholds.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.split import load_split

WINDOWS_PARQUET_PATH = Path("data/features/windows.parquet")
THRESHOLDS_JSON_PATH = Path("models/baseline_thresholds.json")

# Target feature column names in windows.parquet
COL_ANGLE_MAX = "torso_angle_max"
COL_SPEED_MIN = "hip_speed_min"
COL_HEIGHT_MEAN = "hip_height_mean"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate classification performance metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).

    Returns:
        dict containing tp, fp, tn, fn, sensitivity (recall), specificity,
        precision, and f1 score.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        (2.0 * precision * sensitivity) / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }


class BaselineRuleModel:
    """Heuristic rule classifier with scikit-learn compatible predict interface."""

    def __init__(self, A: float, S: float, H: float):
        self.A = float(A)
        self.S = float(S)
        self.H = float(H)

    def predict(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]) -> np.ndarray:
        """Predict fall (1) or non-fall (0) per row.

        Rule:
            torso_angle_max > A AND hip_speed_min < -S AND hip_height_mean < H

        Args:
            X: DataFrame with feature columns, or dict of feature values.

        Returns:
            1D numpy array of int labels (0 or 1).
        """
        if isinstance(X, dict):
            angle = float(X.get(COL_ANGLE_MAX, np.nan))
            speed = float(X.get(COL_SPEED_MIN, np.nan))
            height = float(X.get(COL_HEIGHT_MEAN, np.nan))
            pred = (
                1
                if (angle > self.A and speed < -self.S and height < self.H)
                else 0
            )
            return np.array([pred], dtype=np.int64)

        if isinstance(X, pd.DataFrame):
            angle = X[COL_ANGLE_MAX].to_numpy(dtype=float)
            speed = X[COL_SPEED_MIN].to_numpy(dtype=float)
            height = X[COL_HEIGHT_MEAN].to_numpy(dtype=float)
        else:
            # Assume 2D numpy array where columns correspond to [angle_max, speed_min, height_mean]
            X_arr = np.asarray(X, dtype=float)
            angle = X_arr[:, 0]
            speed = X_arr[:, 1]
            height = X_arr[:, 2]

        cond = (angle > self.A) & (speed < -self.S) & (height < self.H)
        preds = np.where(cond, 1, 0).astype(np.int64)
        return preds


# Global cached model instance for module-level predict()
_DEFAULT_MODEL: Optional[BaselineRuleModel] = None


def load_model(thresholds_path: Union[str, Path] = THRESHOLDS_JSON_PATH) -> BaselineRuleModel:
    """Load trained threshold configuration from JSON."""
    thresholds_path = Path(thresholds_path)
    if not thresholds_path.exists():
        raise FileNotFoundError(
            f"Baseline thresholds file not found at: {thresholds_path}. "
            "Run 'python src/baseline_rules.py' first to tune thresholds."
        )

    with open(thresholds_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return BaselineRuleModel(A=config["A"], S=config["S"], H=config["H"])


def predict(
    window_features_df: Union[pd.DataFrame, Dict[str, Any]],
    thresholds: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Module-level predict function with standard sklearn interface.

    Args:
        window_features_df: DataFrame containing window features.
        thresholds: Optional dict specifying {'A': float, 'S': float, 'H': float}.
                    If None, loads from models/baseline_thresholds.json.

    Returns:
        1D numpy array of predictions (0 or 1).
    """
    global _DEFAULT_MODEL
    if thresholds is not None:
        model = BaselineRuleModel(A=thresholds["A"], S=thresholds["S"], H=thresholds["H"])
        return model.predict(window_features_df)

    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = load_model()

    return _DEFAULT_MODEL.predict(window_features_df)


def tune_baseline_rules(
    windows_path: Union[str, Path] = WINDOWS_PARQUET_PATH,
    output_path: Union[str, Path] = THRESHOLDS_JSON_PATH,
) -> Tuple[BaselineRuleModel, Dict[str, Any]]:
    """Grid search over A, S, H maximizing training set F1."""
    windows_path = Path(windows_path)
    output_path = Path(output_path)

    if not windows_path.exists():
        raise FileNotFoundError(f"Windows dataset not found: {windows_path}")

    df_all = pd.read_parquet(windows_path)

    # 1. Inspect and print feature column names
    print("\n" + "=" * 65)
    print("           INSPECTING WINDOWS DATASET COLUMNS")
    print("=" * 65)
    print(f"Total rows in {windows_path}: {len(df_all)}")
    print(f"Columns present ({len(df_all.columns)}):")
    for col in df_all.columns:
        print(f"  - {col}")
    print("-" * 65)

    # Verify target columns exist
    for req_col in [COL_ANGLE_MAX, COL_SPEED_MIN, COL_HEIGHT_MEAN]:
        assert req_col in df_all.columns, f"Required column '{req_col}' missing from windows dataset!"
    print(f"[OK] Verified presence of rule features: {COL_ANGLE_MAX}, {COL_SPEED_MIN}, {COL_HEIGHT_MEAN}")
    print("=" * 65 + "\n")

    # 2. Load train/test split from src/split.py
    train_ids, test_ids = load_split()
    train_df = df_all[df_all["video_id"].isin(train_ids)].copy()
    test_df = df_all[df_all["video_id"].isin(test_ids)].copy()

    print(f"Training windows: {len(train_df)} from {len(train_ids)} videos")
    print(f"Test windows    : {len(test_df)} from {len(test_ids)} videos")

    y_train = train_df["label"].to_numpy(dtype=int)
    angle_train = train_df[COL_ANGLE_MAX].to_numpy(dtype=float)
    speed_train = train_df[COL_SPEED_MIN].to_numpy(dtype=float)
    height_train = train_df[COL_HEIGHT_MEAN].to_numpy(dtype=float)

    # 3. Grid Search parameter ranges
    # A in 30..80 step 5 (inclusive)
    A_values = [float(a) for a in range(30, 85, 5)]
    # S in 0.5..3.0 step 0.25 (inclusive)
    S_values = [round(float(s), 2) for s in np.arange(0.5, 3.01, 0.25)]
    # H in 0.2..1.2 step 0.1 (inclusive)
    H_values = [round(float(h), 2) for h in np.arange(0.2, 1.21, 0.1)]

    best_f1 = -1.0
    best_combo: Optional[Tuple[float, float, float]] = None
    best_metrics: Dict[str, float] = {}

    total_candidates = len(A_values) * len(S_values) * len(H_values)
    print(f"Executing grid search over {total_candidates} parameter combinations (Training set only)...")

    for A in A_values:
        for S in S_values:
            for H in H_values:
                cond = (angle_train > A) & (speed_train < -S) & (height_train < H)
                preds = np.where(cond, 1, 0)

                metrics = compute_metrics(y_train, preds)
                f1 = metrics["f1"]

                # Maximize F1, breaking ties by sensitivity then specificity
                if (
                    (f1 > best_f1)
                    or (
                        np.isclose(f1, best_f1)
                        and (
                            metrics["sensitivity"] > best_metrics.get("sensitivity", 0.0)
                            or (
                                np.isclose(metrics["sensitivity"], best_metrics.get("sensitivity", 0.0))
                                and metrics["specificity"] > best_metrics.get("specificity", 0.0)
                            )
                        )
                    )
                ):
                    best_f1 = f1
                    best_combo = (A, S, H)
                    best_metrics = metrics

    if best_combo is None:
        raise RuntimeError("Grid search failed to find optimal parameters.")

    opt_A, opt_S, opt_H = best_combo
    best_model = BaselineRuleModel(A=opt_A, S=opt_S, H=opt_H)

    # 4. Save chosen thresholds to JSON
    save_payload = {
        "A": opt_A,
        "S": opt_S,
        "H": opt_H,
        "angle_threshold_deg": opt_A,
        "speed_downward_threshold": opt_S,
        "height_mean_threshold": opt_H,
        "formula": f"{COL_ANGLE_MAX} > {opt_A} AND {COL_SPEED_MIN} < -{opt_S} AND {COL_HEIGHT_MEAN} < {opt_H}",
        "train_metrics": best_metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, indent=2)

    # 5. False positive analysis on training ADL videos
    train_preds = best_model.predict(train_df)
    train_df["pred"] = train_preds
    # Filter for false positive windows from training ADL videos (ground truth label == 0)
    fp_windows = train_df[(train_df["label"] == 0) & (train_df["pred"] == 1)]
    fp_by_video = (
        fp_windows.groupby("video_id")
        .size()
        .reset_index(name="fp_count")
        .sort_values(by=["fp_count", "video_id"], ascending=[False, True])
    )

    # Also list all training ADL videos even if 0 FP windows
    train_adl_ids = [vid for vid in train_ids if "adl" in vid.lower()]
    all_adl_fp = []
    fp_dict = dict(zip(fp_by_video["video_id"], fp_by_video["fp_count"]))
    for vid in train_adl_ids:
        all_adl_fp.append({"video_id": vid, "fp_count": fp_dict.get(vid, 0)})
    all_adl_fp_df = pd.DataFrame(all_adl_fp).sort_values(by=["fp_count", "video_id"], ascending=[False, True])
    top5_adl_fp = all_adl_fp_df.head(5)

    # 6. Evaluate strictly on TEST videos (for reference only, no tuning)
    y_test = test_df["label"].to_numpy(dtype=int)
    test_preds = best_model.predict(test_df)
    test_metrics = compute_metrics(y_test, test_preds)

    # 7. Print formatted summary report
    print("\n" + "=" * 65)
    print("         FALLGUARD BASELINE RULE MODEL - TUNING REPORT")
    print("=" * 65)
    print("Optimal Rule Thresholds (Max Training F1):")
    print(f"  - A (torso_angle_max > A)    : {opt_A:.1f} deg")
    print(f"  - S (hip_speed_min < -S)     : {opt_S:.2f} torso_lengths/s (downward)")
    print(f"  - H (hip_height_mean < H)    : {opt_H:.2f} torso_lengths above floor")
    print(f"  Thresholds saved to          : {output_path}")
    print("-" * 65)
    print("TRAINING METRICS (Optimized On):")
    print(f"  - F1 Score    : {best_metrics['f1']:.4f}")
    print(f"  - Sensitivity : {best_metrics['sensitivity']:.4f} ({best_metrics['tp']}/{best_metrics['tp'] + best_metrics['fn']} falls detected)")
    print(f"  - Specificity : {best_metrics['specificity']:.4f} ({best_metrics['tn']}/{best_metrics['tn'] + best_metrics['fp']} non-falls rejected)")
    print(f"  - Precision   : {best_metrics['precision']:.4f}")
    print(f"  - TP: {best_metrics['tp']} | FP: {best_metrics['fp']} | TN: {best_metrics['tn']} | FN: {best_metrics['fn']}")
    print("-" * 65)
    print("TOP 5 TRAINING ADL VIDEOS WITH MOST FALSE-POSITIVE WINDOWS:")
    for idx, r in top5_adl_fp.iterrows():
        print(f"  {r['video_id']:<10} : {r['fp_count']} false-positive window(s)")
    print("-" * 65)
    print("TEST (for reference only, do not tune on this):")
    print(f"  - F1 Score    : {test_metrics['f1']:.4f}")
    print(f"  - Sensitivity : {test_metrics['sensitivity']:.4f} ({test_metrics['tp']}/{test_metrics['tp'] + test_metrics['fn']} falls detected)")
    print(f"  - Specificity : {test_metrics['specificity']:.4f} ({test_metrics['tn']}/{test_metrics['tn'] + test_metrics['fp']} non-falls rejected)")
    print(f"  - Precision   : {test_metrics['precision']:.4f}")
    print(f"  - TP: {test_metrics['tp']} | FP: {test_metrics['fp']} | TN: {test_metrics['tn']} | FN: {test_metrics['fn']}")
    print("=" * 65 + "\n")

    return best_model, save_payload


def main() -> None:
    tune_baseline_rules()


if __name__ == "__main__":
    main()
