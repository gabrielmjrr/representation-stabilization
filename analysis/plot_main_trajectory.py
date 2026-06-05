"""
analysis/plot_main_trajectory.py — Generate all main trajectory plots from
master_trajectory.csv and cka_matrix.csv.

Reads:
  results/master_trajectory.csv  — produced by analysis/build_master_table.py
  results/cka_matrix.csv         — produced by metrics/cka_trajectory.py

Writes eight figures to config["paths"]["plots"]:
  1.  learning_rate.png
      Learning rate schedule over training epochs.

  2.  accuracy_trajectories.png
      Network test accuracy and all surrogate probe accuracies vs epoch.

  3.  cka_local_change.png
      Per-epoch CKA change (1 − CKA consecutive) with tau reference line.

  4.  cka_to_final.png
      CKA similarity to the final representation vs epoch.

  5.  mean_future_cka.png
      Mean CKA to all future stored representations vs epoch.

  6.  nc1_trajectory.png
      log10(NC1) over training — tracks within/between class scatter ratio.

  7.  cka_heatmap.png
      Epoch × epoch CKA similarity matrix heatmap.

  8.  gap_vs_cka.png
      Logistic probe gap-to-final vs cka_to_final (scatter, colored by epoch).
      Only plotted if all required columns exist.

All plots:
  - Use a non-interactive Agg backend (safe for headless cluster execution).
  - Save as PNG at 150 dpi.
  - Use epoch on the x-axis.
  - Are reproducible (no random state; same data → same figure).

Usage:
  python analysis/plot_main_trajectory.py \\
      --config configs/resnet18_cifar10_200_fullstudy.yaml
"""

import argparse
import csv
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")   # non-interactive; required on cluster (no display)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Logging and config
# ---------------------------------------------------------------------------

def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("plot_main_trajectory")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_master_trajectory(path: str, logger: logging.Logger) -> pd.DataFrame:
    """Load master_trajectory.csv and convert numeric columns."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"master_trajectory.csv not found: {path}\n"
            "Run: python analysis/build_master_table.py --config <config>"
        )
    df = pd.read_csv(path)
    logger.info(f"Loaded master_trajectory.csv: {df.shape}  path={path}")
    return df


def load_cka_matrix(
    path: str,
    logger: logging.Logger,
) -> tuple:
    """
    Load cka_matrix.csv into a numpy matrix and list of epoch labels.

    cka_matrix.csv format (from metrics/cka_trajectory.py):
      First column header: 'epoch' (row epoch label)
      Remaining headers: epoch numbers as strings (column epoch labels)
      Values: CKA floats

    Returns:
        (epoch_labels, cka_matrix_array) or (None, None) if file missing.
    """
    if not os.path.exists(path):
        logger.warning(
            f"cka_matrix.csv not found: {path}. CKA heatmap will be skipped."
        )
        return None, None

    try:
        matrix_df = pd.read_csv(path, index_col="epoch")
        epoch_labels = [int(c) for c in matrix_df.columns]
        cka_array = matrix_df.to_numpy(dtype=float)
        logger.info(
            f"Loaded cka_matrix.csv: {cka_array.shape}  "
            f"epoch range {epoch_labels[0]}–{epoch_labels[-1]}"
        )
        return epoch_labels, cka_array
    except Exception as e:
        logger.error(f"Failed to load cka_matrix.csv: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def col_as_float(df: pd.DataFrame, col: str) -> np.ndarray:
    """Extract a DataFrame column as a float64 numpy array (NaN for missing)."""
    if col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def fig_path(plots_dir: str, name: str) -> str:
    """Return the full path for a figure file."""
    return os.path.join(plots_dir, name)


# ---------------------------------------------------------------------------
# Plot 1: Learning rate schedule
# ---------------------------------------------------------------------------

def plot_learning_rate(
    df: pd.DataFrame,
    plots_dir: str,
    logger: logging.Logger,
) -> str:
    epochs = col_as_float(df, "epoch")
    lr = col_as_float(df, "lr")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, lr, color="steelblue", linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = fig_path(plots_dir, "learning_rate.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Plot 2: Network and probe accuracies
# ---------------------------------------------------------------------------

def plot_accuracy_trajectories(
    df: pd.DataFrame,
    plots_dir: str,
    logger: logging.Logger,
) -> str:
    epochs = col_as_float(df, "epoch")

    probe_specs = [
        ("network_test_acc",  "Network (test)",    "black",      2.5),
        ("logistic_acc",      "Logistic",          "royalblue",  1.5),
        ("linear_svm_acc",    "Linear SVM",        "steelblue",  1.5),
        ("rbf_svm_acc",       "RBF SVM",           "cornflowerblue", 1.5),
        ("rf_acc",            "Random Forest",     "darkorange", 1.5),
        ("lightgbm_acc",      "LightGBM",          "seagreen",   1.5),
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    for col, label, color, lw in probe_specs:
        values = col_as_float(df, col)
        if not np.all(np.isnan(values)):
            ax.plot(epochs, values, label=label, color=color, linewidth=lw,
                    marker="o", markersize=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Network Test Accuracy and Surrogate Probe Accuracies")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = fig_path(plots_dir, "accuracy_trajectories.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Plot 3: Local CKA change with tau reference line
# ---------------------------------------------------------------------------

def plot_local_cka_change(
    df: pd.DataFrame,
    tau: float,
    plots_dir: str,
    logger: logging.Logger,
) -> str:
    epochs = col_as_float(df, "epoch")
    local_change = col_as_float(df, "local_cka_change")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, local_change, color="crimson", linewidth=1.8,
            marker="o", markersize=3, label="local_cka_change")
    ax.axhline(y=tau, color="gray", linestyle="--", linewidth=1.2,
               label=f"tau = {tau}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("1 - CKA(t, t-1)")
    ax.set_title("Local CKA Change with Stabilization Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = fig_path(plots_dir, "cka_local_change.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Plot 4: CKA to final representation
# ---------------------------------------------------------------------------

def plot_cka_to_final(
    df: pd.DataFrame,
    plots_dir: str,
    logger: logging.Logger,
) -> str:
    epochs = col_as_float(df, "epoch")
    cka_to_final = col_as_float(df, "cka_to_final")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, cka_to_final, color="darkorchid", linewidth=1.8,
            marker="o", markersize=3)
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("CKA(t, final)")
    ax.set_title("CKA Similarity to Final Representation")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = fig_path(plots_dir, "cka_to_final.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Plot 5: Mean future CKA
# ---------------------------------------------------------------------------

def plot_mean_future_cka(
    df: pd.DataFrame,
    plots_dir: str,
    logger: logging.Logger,
) -> str:
    epochs = col_as_float(df, "epoch")
    mean_future = col_as_float(df, "mean_future_cka")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, mean_future, color="teal", linewidth=1.8,
            marker="o", markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean CKA to all future checkpoints")
    ax.set_title("Mean Future CKA")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = fig_path(plots_dir, "mean_future_cka.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Plot 6: log10(NC1) trajectory
# ---------------------------------------------------------------------------

def plot_log10_nc1(
    df: pd.DataFrame,
    plots_dir: str,
    logger: logging.Logger,
) -> str:
    epochs = col_as_float(df, "epoch")
    log10_nc1 = col_as_float(df, "log10_nc1")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, log10_nc1, color="saddlebrown", linewidth=1.8,
            marker="o", markersize=3)
    # Reference: log10(NC1) = 0 means NC1 = 1 (within ≈ between scatter)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6,
               label="log10(NC1) = 0")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("log10(NC1)")
    ax.set_title("Neural Collapse — NC1 (log10 scale)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = fig_path(plots_dir, "nc1_trajectory.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Plot 7: CKA heatmap
# ---------------------------------------------------------------------------

def plot_cka_heatmap(
    epoch_labels: list,
    cka_matrix: np.ndarray,
    plots_dir: str,
    logger: logging.Logger,
) -> str:
    n = len(epoch_labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(cka_matrix, vmin=0.0, vmax=1.0, cmap="viridis",
                    aspect="auto", origin="upper")
    plt.colorbar(img, ax=ax, label="CKA similarity")

    # Tick marks: show a subset of epoch labels so they do not overlap
    # Show up to 15 ticks; always show the first and last
    max_ticks = 15
    if n <= max_ticks:
        tick_indices = list(range(n))
    else:
        step = max(1, n // (max_ticks - 1))
        tick_indices = list(range(0, n, step))
        if (n - 1) not in tick_indices:
            tick_indices.append(n - 1)

    tick_labels = [str(epoch_labels[i]) for i in tick_indices]

    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_indices)
    ax.set_yticklabels(tick_labels, fontsize=8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Epoch")
    ax.set_title(f"CKA Similarity Matrix ({n} x {n} checkpoint epochs)")
    fig.tight_layout()

    save_path = fig_path(plots_dir, "cka_heatmap.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Plot 8: Probe gap-to-final versus CKA metrics
# ---------------------------------------------------------------------------

def plot_gap_vs_cka(
    df: pd.DataFrame,
    plots_dir: str,
    logger: logging.Logger,
) -> str:
    """
    Scatter plot: logistic probe gap-to-final vs cka_to_final.
    Each point is one checkpoint epoch.  Points are colored by epoch number.

    Only plotted if the required columns are present.
    """
    required_cols = ["logistic_gap_to_final", "cka_to_final", "epoch"]
    for col in required_cols:
        if col not in df.columns:
            logger.info(f"Skipping gap_vs_cka: column '{col}' not present")
            return ""

    epochs = col_as_float(df, "epoch")
    gap = col_as_float(df, "logistic_gap_to_final")
    cka_to_final = col_as_float(df, "cka_to_final")
    local_change = col_as_float(df, "local_cka_change")

    # Filter to rows where all three values are finite
    valid = np.isfinite(gap) & np.isfinite(cka_to_final) & np.isfinite(epochs)

    if valid.sum() < 2:
        logger.info("Skipping gap_vs_cka: fewer than 2 valid rows")
        return ""

    # Normalize epoch to [0, 1] for coloring
    epoch_min = epochs[valid].min()
    epoch_max = epochs[valid].max()
    if epoch_max == epoch_min:
        epoch_norm = np.zeros(valid.sum())
    else:
        epoch_norm = (epochs[valid] - epoch_min) / (epoch_max - epoch_min)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left subplot: gap vs cka_to_final
    sc = axes[0].scatter(
        cka_to_final[valid], gap[valid],
        c=epoch_norm, cmap="plasma", s=40, alpha=0.8,
    )
    plt.colorbar(sc, ax=axes[0], label="Epoch (normalized)")
    axes[0].set_xlabel("cka_to_final")
    axes[0].set_ylabel("logistic_gap_to_final")
    axes[0].set_title("Probe Gap-to-Final vs CKA-to-Final")
    axes[0].axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    axes[0].grid(True, alpha=0.3)

    # Right subplot: gap vs local_cka_change (if available)
    valid2 = valid & np.isfinite(local_change)
    if valid2.sum() >= 2:
        epoch_norm2 = (epochs[valid2] - epoch_min) / max(epoch_max - epoch_min, 1e-10)
        sc2 = axes[1].scatter(
            local_change[valid2], gap[valid2],
            c=epoch_norm2, cmap="plasma", s=40, alpha=0.8,
        )
        plt.colorbar(sc2, ax=axes[1], label="Epoch (normalized)")
        axes[1].set_xlabel("local_cka_change")
        axes[1].set_ylabel("logistic_gap_to_final")
        axes[1].set_title("Probe Gap-to-Final vs Local CKA Change")
        axes[1].axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].set_visible(False)

    fig.suptitle(
        "Representation Sufficiency vs Stabilization\n"
        "(Each point = one checkpoint epoch; color = epoch progression)",
        fontsize=11,
    )
    fig.tight_layout()

    save_path = fig_path(plots_dir, "gap_vs_cka.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Parse arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Generate all main trajectory plots from master_trajectory.csv. "
            "Run analysis/build_master_table.py first."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/resnet18_cifar10_200_fullstudy.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Override the results directory from config.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="Override the plots output directory from config.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 2. Load config and resolve paths
    # ------------------------------------------------------------------
    config = load_config(args.config)

    results_dir = args.results_dir or config["paths"]["results"]
    plots_dir = args.plots_dir or config["paths"].get(
        "plots", os.path.join(results_dir, "plots")
    )
    logs_dir = config["paths"].get("logs", results_dir)
    tau = config.get("tau", 0.02)

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Logging
    # ------------------------------------------------------------------
    log_path = os.path.join(logs_dir, "plot_main_trajectory.log")
    logger = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("Main Trajectory Plots")
    logger.info("=" * 70)
    logger.info(f"Config:      {args.config}")
    logger.info(f"Results dir: {results_dir}")
    logger.info(f"Plots dir:   {plots_dir}")
    logger.info(f"Final plot output directory: {plots_dir}")
    logger.info(f"tau:         {tau}")

    # ------------------------------------------------------------------
    # 4. Load master trajectory
    # ------------------------------------------------------------------
    master_path = os.path.join(results_dir, "master_trajectory.csv")
    df = load_master_trajectory(master_path, logger)

    # ------------------------------------------------------------------
    # 5. Load CKA matrix (for heatmap)
    # ------------------------------------------------------------------
    cka_matrix_path = os.path.join(results_dir, "cka_matrix.csv")
    epoch_labels, cka_matrix = load_cka_matrix(cka_matrix_path, logger)

    # ------------------------------------------------------------------
    # 6. Generate each plot — wrap each in try/except so one failure
    #    does not prevent the remaining plots from being saved.
    # ------------------------------------------------------------------
    saved_paths = []
    failed_plots = []

    plot_tasks = [
        ("learning_rate",          lambda: plot_learning_rate(df, plots_dir, logger)),
        ("accuracy_trajectories",  lambda: plot_accuracy_trajectories(df, plots_dir, logger)),
        ("cka_local_change",       lambda: plot_local_cka_change(df, tau, plots_dir, logger)),
        ("cka_to_final",           lambda: plot_cka_to_final(df, plots_dir, logger)),
        ("mean_future_cka",        lambda: plot_mean_future_cka(df, plots_dir, logger)),
        ("nc1_trajectory",         lambda: plot_log10_nc1(df, plots_dir, logger)),
        ("cka_heatmap",            lambda: plot_cka_heatmap(epoch_labels, cka_matrix, plots_dir, logger)
                                           if epoch_labels is not None
                                           else logger.info("Skipping cka_heatmap: matrix not loaded")),
        ("gap_vs_cka",             lambda: plot_gap_vs_cka(df, plots_dir, logger)),
    ]

    for plot_name, plot_fn in plot_tasks:
        try:
            result = plot_fn()
            if result:
                saved_paths.append(result)
        except Exception as e:
            logger.error(f"Plot '{plot_name}' failed: {e}")
            failed_plots.append(plot_name)

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info(f"Plot generation complete.")
    logger.info(f"  Saved: {len(saved_paths)} figures to {plots_dir}")
    if failed_plots:
        logger.warning(f"  Failed: {failed_plots}")
    for path in saved_paths:
        logger.info(f"    {os.path.basename(path)}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
