"""plot_eda.py - Exploratory Kinematic Trajectory Visualization.

Purpose:
--------
Generates high-resolution EDA plots showing the temporal dynamics of:
1. Torso Angle (degrees from vertical)
2. Hip Height (normalized by torso length above floor)
3. Hip Speed (vertical derivative, negative = downward)

Visualizes 3 Fall videos (with falling phase shaded) and 3 ADL videos
(including at least one lying ADL such as adl-32).

Output:
-------
Saves figures to `reports/eda/*.png`.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import frame_measurements

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fallguard.plot_eda")

# Plot Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
COLOR_ANGLE = "#0d9488"      # Medical Teal
COLOR_HEIGHT = "#2563eb"     # Royal Blue
COLOR_SPEED = "#d97706"      # Amber
COLOR_FALL_SHADE = "#ef4444" # Alert Red


def plot_single_video(
    video_id: str,
    row: pd.Series,
    landmarks_dir: Path,
    output_dir: Path,
) -> None:
    """Generate and save 3-panel kinematic profile plot for one video."""
    parquet_path = landmarks_dir / f"{video_id}.parquet"
    if not parquet_path.exists():
        logger.warning(f"Landmark parquet missing for {video_id}: {parquet_path}")
        return

    df_lm = pd.read_parquet(parquet_path)
    meas = frame_measurements(df_lm)

    fps = float(row.get("fps", 30.0))
    label = str(row.get("label", "unknown")).upper()
    has_lying = bool(row.get("has_lying_adl", False))

    is_fall = label.lower() == "FALL"
    fall_start_t: Optional[float] = None
    fall_end_t: Optional[float] = None

    if is_fall:
        f_s = row.get("fall_start_frame")
        f_e = row.get("fall_end_frame")
        if pd.notna(f_s) and pd.notna(f_e):
            fall_start_t = max(0.0, (float(f_s) - 1.0) / fps)
            fall_end_t = max(fall_start_t, (float(f_e) - 1.0) / fps)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    t = meas["time_s"].to_numpy()

    # Title & Subtitle
    lying_str = " (Includes Lying Posture)" if (label == "ADL" and has_lying) else ""
    fig.suptitle(
        f"Kinematic Profile: {video_id} [{label}{lying_str}]",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # 1. Torso Angle Subplot
    ax1.plot(t, meas["torso_angle"], color=COLOR_ANGLE, linewidth=2.0, label="Torso Angle (deg)")
    ax1.axhline(0, color="#94a3b8", linestyle="--", linewidth=1.0, alpha=0.7, label="Upright (0°)")
    ax1.axhline(90, color="#dc2626", linestyle=":", linewidth=1.2, alpha=0.7, label="Horizontal (90°)")
    ax1.set_ylabel("Torso Angle (°)", fontsize=11, fontweight="bold")
    ax1.set_ylim(-5, max(105, np.nanmax(meas["torso_angle"]) + 10 if not np.isnan(meas["torso_angle"]).all() else 100))
    ax1.legend(loc="upper left", frameon=True, fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Hip Height Subplot
    ax2.plot(t, meas["hip_height"], color=COLOR_HEIGHT, linewidth=2.0, label="Hip Height (Norm. by Torso)")
    ax2.axhline(0, color="#64748b", linestyle="--", linewidth=1.0, alpha=0.7, label="Floor Level (0.0)")
    ax2.set_ylabel("Hip Height", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Hip Speed Subplot
    ax3.plot(t, meas["hip_speed"], color=COLOR_SPEED, linewidth=2.0, label="Hip Vertical Speed (Units/s)")
    ax3.axhline(0, color="#64748b", linestyle="--", linewidth=1.0, alpha=0.7)
    ax3.set_ylabel("Vertical Speed", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    ax3.legend(loc="lower left", frameon=True, fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # Shade falling phase for falls across all 3 subplots
    if is_fall and (fall_start_t is not None) and (fall_end_t is not None):
        for ax in (ax1, ax2, ax3):
            ax.axvspan(
                fall_start_t,
                fall_end_t,
                color=COLOR_FALL_SHADE,
                alpha=0.22,
                label="Falling Phase (Annotated)" if ax == ax1 else None,
            )
        # Refresh legend for ax1 with shaded region
        ax1.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = output_dir / f"{video_id}_kinematics.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved EDA plot: {out_path}")


def plot_comparison_fall_vs_lying_adl(
    fall_id: str,
    adl_id: str,
    videos_df: pd.DataFrame,
    landmarks_dir: Path,
    output_dir: Path,
) -> None:
    """Generate side-by-side comparative plot: Sudden Fall vs Lying ADL."""
    row_fall = videos_df[videos_df["video_id"] == fall_id].iloc[0]
    row_adl = videos_df[videos_df["video_id"] == adl_id].iloc[0]

    meas_fall = frame_measurements(pd.read_parquet(landmarks_dir / f"{fall_id}.parquet"))
    meas_adl = frame_measurements(pd.read_parquet(landmarks_dir / f"{adl_id}.parquet"))

    fps_fall = float(row_fall.get("fps", 30.0))
    f_s = row_fall.get("fall_start_frame")
    f_e = row_fall.get("fall_end_frame")
    fall_start_t = (float(f_s) - 1.0) / fps_fall if pd.notna(f_s) else None
    fall_end_t = (float(f_e) - 1.0) / fps_fall if pd.notna(f_e) else None

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharey="row")
    fig.suptitle(
        f"Kinematic Comparison: Fall ({fall_id}) vs Lying ADL ({adl_id})",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    datasets = [(meas_fall, axes[:, 0], f"Fall Event: {fall_id}", True),
                (meas_adl, axes[:, 1], f"Controlled Lying ADL: {adl_id}", False)]

    for meas, (ax1, ax2, ax3), col_title, is_f in datasets:
        t = meas["time_s"].to_numpy()
        ax1.set_title(col_title, fontsize=12, fontweight="bold", pad=8)

        # Torso Angle
        ax1.plot(t, meas["torso_angle"], color=COLOR_ANGLE, lw=2.0)
        ax1.axhline(0, color="#94a3b8", ls="--", lw=1.0)
        ax1.axhline(90, color="#dc2626", ls=":", lw=1.2)
        ax1.grid(True, ls="--", alpha=0.5)

        # Hip Height
        ax2.plot(t, meas["hip_height"], color=COLOR_HEIGHT, lw=2.0)
        ax2.axhline(0, color="#64748b", ls="--", lw=1.0)
        ax2.grid(True, ls="--", alpha=0.5)

        # Hip Speed
        ax3.plot(t, meas["hip_speed"], color=COLOR_SPEED, lw=2.0)
        ax3.axhline(0, color="#64748b", ls="--", lw=1.0)
        ax3.set_xlabel("Time (seconds)", fontsize=10, fontweight="bold")
        ax3.grid(True, ls="--", alpha=0.5)

        if is_f and (fall_start_t is not None) and (fall_end_t is not None):
            for ax in (ax1, ax2, ax3):
                ax.axvspan(fall_start_t, fall_end_t, color=COLOR_FALL_SHADE, alpha=0.22)

    axes[0, 0].set_ylabel("Torso Angle (°)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Hip Height", fontsize=10, fontweight="bold")
    axes[2, 0].set_ylabel("Vertical Speed", fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = output_dir / "fall_vs_lying_adl_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved side-by-side comparison EDA plot: {out_path}")


def run_eda(
    catalog_path: Path = Path("data/videos.csv"),
    landmarks_dir: Path = Path("data/landmarks"),
    output_dir: Path = Path("reports/eda"),
) -> None:
    """Run EDA plotting for 3 fall videos and 3 ADL videos (including a lying ADL)."""
    if not catalog_path.exists():
        raise FileNotFoundError(f"Videos catalog not found: {catalog_path}")

    videos_df = pd.read_csv(catalog_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select 3 Fall videos
    fall_ids = ["fall-01", "fall-02", "fall-05"]

    # Select 3 ADL videos, including one lying ADL (adl-32)
    adl_ids = ["adl-01", "adl-07", "adl-32"]

    selected_ids = fall_ids + adl_ids
    logger.info(f"Generating kinematic EDA plots for: {selected_ids}")

    for vid in selected_ids:
        matching = videos_df[videos_df["video_id"] == vid]
        if matching.empty:
            logger.warning(f"Video {vid} not in catalog; skipping.")
            continue
        row = matching.iloc[0]
        plot_single_video(vid, row, landmarks_dir, output_dir)

    # Side-by-side comparative diagnostic (fall-01 vs adl-32)
    plot_comparison_fall_vs_lying_adl("fall-01", "adl-32", videos_df, landmarks_dir, output_dir)

    print("\n" + "=" * 65)
    print("           FALLGUARD EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 65)
    print(f"Generated {len(selected_ids)} individual kinematic profile plots:")
    for vid in selected_ids:
        print(f"  - reports/eda/{vid}_kinematics.png")
    print(f"  - reports/eda/fall_vs_lying_adl_comparison.png")
    print(f"Destination: {output_dir.resolve()}")
    print("=" * 65 + "\n")



if __name__ == "__main__":
    run_eda()
