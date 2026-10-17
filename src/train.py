"""train.py - Cross-Validation, Tuning, and Model Training Pipeline.

Features:
---------
- Uses 2.0-second window statistical features from `data/features/windows.parquet`.
- Strictly adheres to the train/test video split from `src.split.load_split()`.
- Imputes missing feature values with the training-set medians.
- Evaluates models using StratifiedGroupKFold (5 splits, grouped by video_id) to prevent
  any temporal or subject leakage across folds.
- Compares:
  1. Random Forest (balanced_subsample, n_jobs=-1)
  2. XGBoost (scale_pos_weight = n_neg / n_pos, eval_metric="logloss")
  3. Heuristic Baseline Rules (src.baseline_rules.predict)
- Tunes decision probability thresholds on out-of-fold predictions requiring specificity >= 0.95.
- Fits final models on all training videos and exports bundles:
  `models/rf.joblib` and `models/xgb.joblib`.
"""

import itertools
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from src.baseline_rules import predict as baseline_predict
from src.features import STRIDE_S, WINDOW_S
from src.split import load_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fallguard.train")

WINDOWS_PARQUET_PATH = Path("data/features/windows.parquet")
MODELS_DIR = Path("models")
RF_MODEL_PATH = MODELS_DIR / "rf.joblib"
XGB_MODEL_PATH = MODELS_DIR / "xgb.joblib"

# Columns to exclude from feature matrix
DROP_COLS = ["video_id", "dataset", "window_start", "window_end", "label"]


def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculate sensitivity, specificity, precision, F1, and ROC-AUC for a fold."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        (2.0 * precision * sensitivity) / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = 0.5
    elif len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_pred))
        except ValueError:
            auc = 0.5
    else:
        auc = 0.5

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "roc_auc": auc,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    oof_probs: np.ndarray,
    min_specificity: float = 0.95,
) -> Tuple[float, Dict[str, float]]:
    """Find probability threshold maximizing F1 while requiring specificity >= min_specificity."""
    best_t = 0.5
    best_f1 = -1.0
    best_metrics: Dict[str, float] = {}

    threshold_candidates = np.linspace(0.05, 0.95, 91)

    # First pass: thresholds meeting min_specificity
    for t in threshold_candidates:
        preds = (oof_probs >= t).astype(int)
        metrics = compute_fold_metrics(y_true, preds, oof_probs)

        if metrics["specificity"] >= min_specificity:
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_t = float(t)
                best_metrics = metrics

    # Fallback if no threshold meets specificity constraint
    if best_f1 < 0.0:
        logger.warning(f"No threshold achieved specificity >= {min_specificity}. Selecting highest specificity threshold.")
        highest_spec = -1.0
        for t in threshold_candidates:
            preds = (oof_probs >= t).astype(int)
            metrics = compute_fold_metrics(y_true, preds, oof_probs)
            if metrics["specificity"] > highest_spec:
                highest_spec = metrics["specificity"]
                best_t = float(t)
                best_metrics = metrics

    return best_t, best_metrics


def train_and_evaluate() -> None:
    """Execute hyperparameter tuning, cross-validation, threshold optimization, and final training."""
    if not WINDOWS_PARQUET_PATH.exists():
        raise FileNotFoundError(f"Windows dataset not found at: {WINDOWS_PARQUET_PATH}")

    df_all = pd.read_parquet(WINDOWS_PARQUET_PATH)

    # 1. Load split and strictly isolate TRAINING data
    train_ids, test_ids = load_split()
    train_df = df_all[df_all["video_id"].isin(train_ids)].copy().reset_index(drop=True)

    # Step 5: Data Leakage Verification Check
    overlap_videos = set(train_ids) & set(test_ids)
    assert len(overlap_videos) == 0, f"Data leakage error: overlapping videos {overlap_videos}"
    test_windows_in_train = df_all[df_all["video_id"].isin(test_ids) & df_all["video_id"].isin(train_df["video_id"])]
    assert len(test_windows_in_train) == 0, "Test windows leaked into training set!"

    print("\n" + "=" * 65)
    print("                    DATA LEAKAGE AUDIT CHECK")
    print("=" * 65)
    print(f"Train Video IDs Count  : {len(train_ids)}")
    print(f"Test Video IDs Count   : {len(test_ids)}")
    print(f"Overlapping Videos     : {len(overlap_videos)} (Zero overlap confirmed)")
    print(f"Total Windows in Train : {len(train_df)}")
    print(f"Test Windows in Train  : 0 (Zero leakage confirmed)")
    print("Fitting Data Scope     : Strictly 100% TRAIN video windows only")
    print("[PASS] Leakage audit fully passed.")
    print("=" * 65 + "\n")

    # Define ordered feature names (excluding metadata and label)
    feature_names = [col for col in df_all.columns if col not in DROP_COLS]
    print(f"Active Features ({len(feature_names)}):")
    for idx, f in enumerate(feature_names, start=1):
        print(f"  {idx:2d}. {f}")
    print("-" * 65)

    # Compute training-set medians and impute any NaNs
    medians = train_df[feature_names].median().to_dict()
    train_df[feature_names] = train_df[feature_names].fillna(medians)

    X_train = train_df[feature_names].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy(dtype=int)
    groups_train = train_df["video_id"].to_numpy()

    # Setup StratifiedGroupKFold (5 splits, grouped by video_id)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(sgkf.split(X_train, y_train, groups=groups_train))

    # =========================================================================
    # Hyperparameter Search: Random Forest
    # =========================================================================
    print("\nTuning Random Forest hyperparameters over 5 Stratified Group folds...")
    rf_grid = {
        "n_estimators": [200, 400],
        "max_depth": [3, 5, 8],
        "min_samples_leaf": [2, 5],
    }

    best_rf_f1 = -1.0
    best_rf_params: Dict[str, Any] = {}

    rf_combos = list(itertools.product(
        rf_grid["n_estimators"],
        rf_grid["max_depth"],
        rf_grid["min_samples_leaf"],
    ))

    for n_est, m_depth, m_leaf in rf_combos:
        fold_f1s = []
        for tr_idx, val_idx in folds:
            clf = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=m_depth,
                min_samples_leaf=m_leaf,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42,
            )
            clf.fit(X_train[tr_idx], y_train[tr_idx])
            preds = clf.predict(X_train[val_idx])
            fold_f1s.append(f1_score(y_train[val_idx], preds, zero_division=0))

        mean_f1 = float(np.mean(fold_f1s))
        if mean_f1 > best_rf_f1:
            best_rf_f1 = mean_f1
            best_rf_params = {
                "n_estimators": n_est,
                "max_depth": m_depth,
                "min_samples_leaf": m_leaf,
            }

    print(f"[OK] Best Random Forest Parameters: {best_rf_params} (Mean CV F1: {best_rf_f1:.4f})")

    # =========================================================================
    # Hyperparameter Search: XGBoost
    # =========================================================================
    print("Tuning XGBoost hyperparameters over 5 Stratified Group folds...")
    xgb_grid = {
        "n_estimators": [100, 200],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8],
    }

    best_xgb_f1 = -1.0
    best_xgb_params: Dict[str, Any] = {}

    xgb_combos = list(itertools.product(
        xgb_grid["n_estimators"],
        xgb_grid["max_depth"],
        xgb_grid["learning_rate"],
        xgb_grid["subsample"],
    ))

    for n_est, m_depth, lr, subs in xgb_combos:
        fold_f1s = []
        for tr_idx, val_idx in folds:
            n_neg = float(np.sum(y_train[tr_idx] == 0))
            n_pos = float(np.sum(y_train[tr_idx] == 1))
            spw = n_neg / n_pos if n_pos > 0 else 1.0

            clf = XGBClassifier(
                n_estimators=n_est,
                max_depth=m_depth,
                learning_rate=lr,
                subsample=subs,
                scale_pos_weight=spw,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
            clf.fit(X_train[tr_idx], y_train[tr_idx])
            preds = clf.predict(X_train[val_idx])
            fold_f1s.append(f1_score(y_train[val_idx], preds, zero_division=0))

        mean_f1 = float(np.mean(fold_f1s))
        if mean_f1 > best_xgb_f1:
            best_xgb_f1 = mean_f1
            best_xgb_params = {
                "n_estimators": n_est,
                "max_depth": m_depth,
                "learning_rate": lr,
                "subsample": subs,
            }

    print(f"[OK] Best XGBoost Parameters      : {best_xgb_params} (Mean CV F1: {best_xgb_f1:.4f})")

    # =========================================================================
    # 5-Fold Cross Validation Evaluation of Best Models & Rule Baseline
    # =========================================================================
    print("\nRunning full 5-fold cross-validation with optimal parameters...")

    rf_fold_metrics: List[Dict[str, float]] = []
    xgb_fold_metrics: List[Dict[str, float]] = []
    rule_fold_metrics: List[Dict[str, float]] = []

    oof_probs_rf = np.zeros(len(y_train), dtype=float)
    oof_probs_xgb = np.zeros(len(y_train), dtype=float)

    for fold_num, (tr_idx, val_idx) in enumerate(folds, start=1):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        val_sub_df = train_df.iloc[val_idx]

        # 1. Random Forest
        rf_clf = RandomForestClassifier(
            **best_rf_params,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
        rf_clf.fit(X_tr, y_tr)
        rf_probs = rf_clf.predict_proba(X_val)[:, 1]
        oof_probs_rf[val_idx] = rf_probs
        rf_preds = rf_clf.predict(X_val)
        rf_fold_metrics.append(compute_fold_metrics(y_val, rf_preds, rf_probs))

        # 2. XGBoost
        n_neg = float(np.sum(y_tr == 0))
        n_pos = float(np.sum(y_tr == 1))
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        xgb_clf = XGBClassifier(
            **best_xgb_params,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        xgb_clf.fit(X_tr, y_tr)
        xgb_probs = xgb_clf.predict_proba(X_val)[:, 1]
        oof_probs_xgb[val_idx] = xgb_probs
        xgb_preds = xgb_clf.predict(X_val)
        xgb_fold_metrics.append(compute_fold_metrics(y_val, xgb_preds, xgb_probs))

        # 3. Rule Baseline (src.baseline_rules)
        rule_preds = baseline_predict(val_sub_df)
        rule_fold_metrics.append(compute_fold_metrics(y_val, rule_preds, None))

    # Helper function to compute mean ± std
    def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
        agg: Dict[str, Tuple[float, float]] = {}
        for key in ["sensitivity", "specificity", "precision", "f1", "roc_auc"]:
            vals = [m[key] for m in metrics_list]
            agg[key] = (float(np.mean(vals)), float(np.std(vals)))
        return agg

    rf_agg = aggregate_metrics(rf_fold_metrics)
    xgb_agg = aggregate_metrics(xgb_fold_metrics)
    rule_agg = aggregate_metrics(rule_fold_metrics)

    print("\n" + "=" * 78)
    print("    5-FOLD STRATIFIED GROUP CV PERFORMANCE (MEAN +/- STD ACROSS FOLDS)")
    print("=" * 78)
    header = f"{'Metric':<14} | {'Rule Baseline':<18} | {'Random Forest':<18} | {'XGBoost':<18}"
    print(header)
    print("-" * 78)

    for metric in ["sensitivity", "specificity", "precision", "f1", "roc_auc"]:
        r_m, r_s = rule_agg[metric]
        rf_m, rf_s = rf_agg[metric]
        xgb_m, xgb_s = xgb_agg[metric]
        m_name = metric.capitalize() if metric != "roc_auc" else "ROC-AUC"
        print(f"{m_name:<14} | {r_m:.4f} +/- {r_s:.4f}  | {rf_m:.4f} +/- {rf_s:.4f}  | {xgb_m:.4f} +/- {xgb_s:.4f}")
    print("=" * 78 + "\n")

    # =========================================================================
    # Step 3: Choose Probability Decision Threshold on OOF (Train Videos Only)
    # =========================================================================
    print("Optimizing probability decision threshold on out-of-fold predictions (Specificity >= 0.95)...")
    rf_thresh, rf_thresh_metrics = find_optimal_threshold(y_train, oof_probs_rf, min_specificity=0.95)
    xgb_thresh, xgb_thresh_metrics = find_optimal_threshold(y_train, oof_probs_xgb, min_specificity=0.95)

    print(f"  - Random Forest Threshold : {rf_thresh:.3f}")
    print(f"      OOF F1: {rf_thresh_metrics['f1']:.4f} | Sens: {rf_thresh_metrics['sensitivity']:.4f} | Spec: {rf_thresh_metrics['specificity']:.4f} | Prec: {rf_thresh_metrics['precision']:.4f}")
    print(f"  - XGBoost Threshold       : {xgb_thresh:.3f}")
    print(f"      OOF F1: {xgb_thresh_metrics['f1']:.4f} | Sens: {xgb_thresh_metrics['sensitivity']:.4f} | Spec: {xgb_thresh_metrics['specificity']:.4f} | Prec: {xgb_thresh_metrics['precision']:.4f}")

    # =========================================================================
    # Step 4: Fit Final Models on ALL Train Videos & Save Artifacts
    # =========================================================================
    print("\nFitting final models on 100% of training video data...")

    # Final Random Forest
    final_rf = RandomForestClassifier(
        **best_rf_params,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    final_rf.fit(X_train, y_train)

    rf_bundle = {
        "model": final_rf,
        "feature_names": feature_names,
        "medians": medians,
        "threshold": rf_thresh,
        "window_s": float(WINDOW_S),
        "stride_s": float(STRIDE_S),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_bundle, RF_MODEL_PATH)
    logger.info(f"Saved Random Forest model bundle to: {RF_MODEL_PATH}")

    # Final XGBoost
    total_neg = float(np.sum(y_train == 0))
    total_pos = float(np.sum(y_train == 1))
    final_spw = total_neg / total_pos if total_pos > 0 else 1.0

    final_xgb = XGBClassifier(
        **best_xgb_params,
        scale_pos_weight=final_spw,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    final_xgb.fit(X_train, y_train)

    xgb_bundle = {
        "model": final_xgb,
        "feature_names": feature_names,
        "medians": medians,
        "threshold": xgb_thresh,
        "window_s": float(WINDOW_S),
        "stride_s": float(STRIDE_S),
    }

    joblib.dump(xgb_bundle, XGB_MODEL_PATH)
    logger.info(f"Saved XGBoost model bundle to: {XGB_MODEL_PATH}")

    # =========================================================================
    # Step 6: Print Top 10 Feature Importances
    # =========================================================================
    print("\n" + "=" * 65)
    print("                TOP 10 FEATURE IMPORTANCES")
    print("=" * 65)

    print("Random Forest Top 10 Features:")
    rf_imp = final_rf.feature_importances_
    rf_top_idx = np.argsort(rf_imp)[::-1][:10]
    for rank, idx in enumerate(rf_top_idx, start=1):
        print(f"  {rank:2d}. {feature_names[idx]:<22} : {rf_imp[idx]:.4f} ({rf_imp[idx]*100:.1f}%)")

    print("\nXGBoost Top 10 Features:")
    xgb_imp = final_xgb.feature_importances_
    xgb_top_idx = np.argsort(xgb_imp)[::-1][:10]
    for rank, idx in enumerate(xgb_top_idx, start=1):
        print(f"  {rank:2d}. {feature_names[idx]:<22} : {xgb_imp[idx]:.4f} ({xgb_imp[idx]*100:.1f}%)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    train_and_evaluate()
