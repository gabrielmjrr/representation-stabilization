"""
analysis/plot_cka_heatmap.py — Epoch × epoch CKA heatmap (Sharon & Dar 2024 style).

Visualises the full pairwise CKA matrix over training epochs.
Each cell (i, j) shows CKA(epoch_i, epoch_j).
Diagonal = 1.0 (a representation is identical to itself).
Off-diagonal blocks of high similarity reveal the three training phases:
  Phase I  — rapid representation change: low off-diagonal similarity
  Phase II — slower / memorisation phase: moderate off-diagonal similarity
  Phase III— post-zero-error: high similarity across all late epochs

Requires:
    results/cka_pairwise_results.csv   — produced by:
        python metrics/cka.py --mode pairwise --config configs/resnet18_cifar10.yaml

Optionally reads:
    results/stabilization_summary.csv  — to overlay t* lines on both axes

Writes:
    results/plot_cka_heatmap.png

Usage:
    python analysis/plot_cka_heatmap.py --config configs/resnet18_cifar10.yaml
    python analysis/plot_cka_heatmap.py --results-dir /path/to/results
"""

import argparse
import csv
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; required on cluster (no display)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Figure style constants
# ---------------------------------------------------------------------------

FIGURE_SIZE_INCHES = (9, 8)
DPI                = 150

# viridis: perceptually uniform, colorblind-safe, dark=dissimilar, bright=similar
COLORMAP = "viridis"

COLOR_TSTAR        = "#ff4444"   # red for t* lines
COLOR_TSTAR_TEXT   = "#cc0000"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("plot_cka_heatmap")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pairwise_cka(results_dir: str) -> list[dict]:
    """
    Load cka_pairwise_results.csv.
    Expected columns: epoch_a, epoch_b, cka_value.

    The file stores the upper triangle only (epoch_a <= epoch_b).
    The full symmetric matrix is reconstructed in build_cka_matrix().
    """
    path = os.path.join(results_dir, "cka_pairwise_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pairwise CKA results not found at '{path}'.\n"
            "Generate them first with:\n"
            "    python metrics/cka.py --mode pairwise --config configs/resnet18_cifar10.yaml"
        )

    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "epoch_a":   int(row["epoch_a"]),
                "epoch_b":   int(row["epoch_b"]),
                "cka_value": float(row["cka_value"]),
            })
    return rows


def load_stabilization_summary(results_dir: str) -> dict:
    """
    Load stabilization_summary.csv.
    Returns dict mapping "CKA" / "DRS" / "JOINT" to t* (int) or None.
    """
    path = os.path.join(results_dir, "stabilization_summary.csv")
    if not os.path.exists(path):
        return {}

    summary = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric     = row["metric"]
            t_star_raw = row["t_star"]
            if t_star_raw in ("NOT_DETECTED", ""):
                t_star = None
            else:
                t_star = int(t_star_raw)
            summary[metric] = t_star

    return summary


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------

def build_cka_matrix(
    pairwise_rows: list[dict],
) -> tuple[np.ndarray, list[int]]:
    """
    Reconstruct the full symmetric CKA matrix from upper-triangle rows.

    Returns:
        cka_matrix  — shape (n_epochs, n_epochs), values in [0, 1]
        epoch_list  — sorted list of epoch integers (axis labels)

    The (i, j) entry of cka_matrix equals CKA(epoch_list[i], epoch_list[j]).
    """
    # Collect all unique epochs from both columns
    epoch_set = set()
    for row in pairwise_rows:
        epoch_set.add(row["epoch_a"])
        epoch_set.add(row["epoch_b"])

    epoch_list = sorted(epoch_set)
    n_epochs   = len(epoch_list)

    # Map epoch number → matrix index for O(1) lookup
    epoch_to_index = {epoch: idx for idx, epoch in enumerate(epoch_list)}

    # Initialise matrix; NaN marks cells that were not provided (should not happen)
    cka_matrix = np.full((n_epochs, n_epochs), fill_value=np.nan)

    for row in pairwise_rows:
        idx_a = epoch_to_index[row["epoch_a"]]
        idx_b = epoch_to_index[row["epoch_b"]]
        value = row["cka_value"]

        # Fill both (a, b) and (b, a) because CKA is symmetric
        cka_matrix[idx_a, idx_b] = value
        cka_matrix[idx_b, idx_a] = value

    n_nan = int(np.isnan(cka_matrix).sum())
    if n_nan > 0:
        raise ValueError(
            f"{n_nan} cells in the CKA matrix are NaN. "
            "The pairwise CSV may be incomplete — re-run metrics/cka.py --mode pairwise."
        )

    return cka_matrix, epoch_list


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cka_heatmap(
    cka_matrix: np.ndarray,
    epoch_list: list[int],
    summary: dict,
    tau: float,
    K: int,
    output_path: str,
    logger: logging.Logger,
) -> None:
    """
    Render the epoch × epoch CKA heatmap and save to output_path.

    Axes: x = epoch_b (horizontal), y = epoch_a (vertical).
    Colorbar: CKA value in [0, 1]; bright = similar, dark = dissimilar.
    t* lines: drawn on both axes when detected.

    Note on axis orientation: epoch 0 at the top-left (origin) keeps the
    temporal reading direction consistent with a narrative ("early" → "late").
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_INCHES)

    # --- Heatmap ---
    im = ax.imshow(
        cka_matrix,
        aspect="equal",
        origin="upper",   # epoch_list[0] at the top-left corner
        cmap=COLORMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    # --- Colorbar ---
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("CKA  (1 = identical, 0 = maximally different)", fontsize=10)

    # --- Axis ticks ---
    # Show one tick per 10 epochs to keep the axis readable
    n_epochs       = len(epoch_list)
    tick_spacing   = max(1, n_epochs // 20)   # aim for ~20 ticks at most
    tick_indices   = list(range(0, n_epochs, tick_spacing))
    tick_labels    = [str(epoch_list[i]) for i in tick_indices]

    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_indices)
    ax.set_yticklabels(tick_labels, fontsize=8)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Epoch", fontsize=11)

    # --- t* overlay lines ---
    # Draw both a vertical and a horizontal line so the full row and column
    # of the stabilization epoch are visible on both axes.
    t_star_cka  = summary.get("CKA",   None)
    t_star_drs  = summary.get("DRS",   None)
    t_star_joint = summary.get("JOINT", None)

    epoch_to_index = {epoch: idx for idx, epoch in enumerate(epoch_list)}

    def draw_tstar_lines(t_star: int, linestyle: str, label: str) -> None:
        if t_star not in epoch_to_index:
            return
        idx = epoch_to_index[t_star]
        ax.axvline(idx, color=COLOR_TSTAR, linestyle=linestyle, linewidth=1.8,
                   label=label, zorder=5)
        ax.axhline(idx, color=COLOR_TSTAR, linestyle=linestyle, linewidth=1.8,
                   zorder=5)
        # Annotate the tick on the x-axis
        ax.text(idx, n_epochs + 0.5, f"t*={t_star}",
                color=COLOR_TSTAR_TEXT, fontsize=7, ha="center", va="bottom",
                clip_on=False)

    if t_star_joint is not None:
        draw_tstar_lines(t_star_joint, linestyle="-",  label=f"t* (joint) = {t_star_joint}")
    elif t_star_cka is not None:
        draw_tstar_lines(t_star_cka,   linestyle="--", label=f"t* (CKA) = {t_star_cka}")
    elif t_star_drs is not None:
        draw_tstar_lines(t_star_drs,   linestyle=":",  label=f"t* (DRS) = {t_star_drs}")

    if any(v is not None for v in [t_star_joint, t_star_cka, t_star_drs]):
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

    # --- Title ---
    tau_label = f"τ = {tau},  K = {K}"
    ax.set_title(
        f"Epoch × Epoch CKA Heatmap  ({tau_label})",
        fontsize=12, fontweight="bold", pad=12,
    )

    # --- Statistics annotation ---
    cka_min  = float(np.nanmin(cka_matrix))
    cka_max  = float(np.nanmax(cka_matrix))
    # Off-diagonal minimum: the most dissimilar epoch pair in training
    off_diag = cka_matrix.copy()
    np.fill_diagonal(off_diag, np.nan)
    off_diag_min = float(np.nanmin(off_diag))

    stats_text = (
        f"min (off-diag) = {off_diag_min:.3f}\n"
        f"max = {cka_max:.3f}"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=7, color="#444444",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the epoch × epoch CKA heatmap. "
            "Requires cka_pairwise_results.csv produced by "
            "'python metrics/cka.py --mode pairwise'."
        )
    )
    parser.add_argument(
        "--config", type=str, default="configs/resnet18_cifar10.yaml",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Override results directory from config.",
    )
    args = parser.parse_args()

    logger = setup_logging()

    config      = load_config(args.config)
    results_dir = args.results_dir or config["paths"]["results"]
    tau         = float(config["tau"])
    K           = int(config["K"])

    logger.info("=" * 60)
    logger.info("plot_cka_heatmap.py — Epoch × Epoch CKA Heatmap")
    logger.info("=" * 60)
    logger.info(f"Results dir : {results_dir}")
    logger.info(f"tau = {tau},  K = {K}")

    pairwise_rows = load_pairwise_cka(results_dir)
    logger.info(f"Loaded {len(pairwise_rows)} pairwise CKA values")

    cka_matrix, epoch_list = build_cka_matrix(pairwise_rows)
    logger.info(
        f"Matrix shape: {cka_matrix.shape}  "
        f"(epochs {epoch_list[0]} – {epoch_list[-1]})"
    )

    summary = load_stabilization_summary(results_dir)
    if summary:
        logger.info(f"Stabilization summary: { {k: v for k, v in summary.items()} }")
    else:
        logger.warning(
            "Stabilization summary not found — t* lines will be omitted. "
            "Run 'python metrics/stabilization.py' to add them."
        )

    output_path = os.path.join(results_dir, "plot_cka_heatmap.png")

    plot_cka_heatmap(
        cka_matrix=cka_matrix,
        epoch_list=epoch_list,
        summary=summary,
        tau=tau,
        K=K,
        output_path=output_path,
        logger=logger,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
