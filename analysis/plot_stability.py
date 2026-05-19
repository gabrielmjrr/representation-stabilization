"""
analysis/plot_stability.py — Plot CKA and DRS temporal stability curves with t* marked.

Reads (from results_dir):
    cka_results.csv           — consecutive-pair CKA  (metrics/cka.py)
    drs_results.csv           — consecutive-pair DRS  (metrics/drs.py)   [optional]
    stabilization_summary.csv — detected t* epochs    (metrics/stabilization.py)

Writes (to results_dir):
    plot_stability.png

Each subplot shows the temporal change metric (1 − similarity) over training epoch.
The tau threshold is drawn as a horizontal reference line.
The detected t* epoch is marked as a vertical line with annotation.
The streak of below-tau pairs that triggered t* detection is shaded.

Usage:
    python analysis/plot_stability.py --config configs/resnet18_cifar10.yaml
    python analysis/plot_stability.py --results-dir /path/to/results
"""

import argparse
import csv
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; required on cluster (no display)
import matplotlib.pyplot as plt
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Figure style constants — all visual parameters here, never inline
# ---------------------------------------------------------------------------

FIGURE_WIDTH_INCHES  = 10
FIGURE_HEIGHT_ONE_PANEL  = 5
FIGURE_HEIGHT_TWO_PANELS = 8
DPI = 150

COLOR_CKA    = "#2166ac"   # blue
COLOR_DRS    = "#d6604d"   # red-orange
COLOR_TAU    = "#404040"   # dark grey
COLOR_TSTAR  = "#1a9641"   # green
COLOR_STABLE = "#a6dba0"   # light green (below-tau region shading)

MARKER_SIZE  = 4
LINE_WIDTH   = 1.6


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("plot_stability")
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

def load_cka_results(results_dir: str) -> list[dict]:
    """
    Load cka_results.csv.
    Returns list of dicts with keys: epoch_prev, epoch_curr, cka_value, cka_change, below_tau.
    """
    path = os.path.join(results_dir, "cka_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CKA results not found at '{path}'. Run 'python metrics/cka.py' first."
        )

    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "epoch_prev": int(row["epoch_prev"]),
                "epoch_curr": int(row["epoch_curr"]),
                "cka_value":  float(row["cka_value"]),
                "cka_change": float(row["cka_change"]),
                "below_tau":  int(row["below_tau"]),
            })
    return rows


def load_drs_results(results_dir: str) -> list[dict]:
    """
    Load drs_results.csv.
    Returns empty list if the file does not exist (DRS is optional).
    """
    path = os.path.join(results_dir, "drs_results.csv")
    if not os.path.exists(path):
        return []

    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "epoch_prev": int(row["epoch_prev"]),
                "epoch_curr": int(row["epoch_curr"]),
                "drs_value":  float(row["drs_value"]),
                "drs_change": float(row["drs_change"]),
                "below_tau":  int(row["below_tau"]),
            })
    return rows


def load_stabilization_summary(results_dir: str) -> dict:
    """
    Load stabilization_summary.csv.
    Returns dict mapping metric name ("CKA", "DRS", "JOINT") to t* (int) or None.
    """
    path = os.path.join(results_dir, "stabilization_summary.csv")
    if not os.path.exists(path):
        return {}

    summary = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric    = row["metric"]
            t_star_raw = row["t_star"]

            if t_star_raw in ("NOT_DETECTED", ""):
                t_star = None
            else:
                t_star = int(t_star_raw)

            summary[metric] = t_star

    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_streak_start(rows: list[dict], key_below_tau: str) -> int | None:
    """
    Find the epoch_curr where the final consecutive below-tau streak began.

    Walk forward through rows. Track the current run. After the loop, if we
    ended in a run, return the epoch_curr of the first row in that run.
    Used to shade the stable region from streak-start to end of training.
    """
    run_start_epoch = None
    in_run = False

    for row in rows:
        if row[key_below_tau] == 1:
            if not in_run:
                in_run = True
                run_start_epoch = row["epoch_curr"]
        else:
            in_run = False
            run_start_epoch = None

    return run_start_epoch if in_run else None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def add_stability_panel(
    ax,
    epochs: list[int],
    changes: list[float],
    below_tau_flags: list[int],
    color: str,
    marker: str,
    metric_label: str,
    tau: float,
    t_star: int | None,
    epoch_end: int,
) -> None:
    """
    Draw a single stability panel (one subplot) in-place on ax.

    Draws:
      - the change curve (1 - similarity) with markers
      - horizontal tau threshold line
      - shaded region from start of stable streak to epoch_end
      - vertical t* line with epoch annotation (or "not detected" text)
    """
    # --- Curve ---
    ax.plot(
        epochs, changes,
        color=color, linewidth=LINE_WIDTH, marker=marker,
        markersize=MARKER_SIZE, label=f"{metric_label} change  (1 − {metric_label})",
        zorder=3,
    )

    # --- Tau threshold ---
    ax.axhline(
        tau, color=COLOR_TAU, linestyle="--", linewidth=1.2,
        label=f"τ = {tau}  (stability threshold)",
        zorder=2,
    )

    # --- Below-tau shading ---
    # Shade each individual checkpoint window that fell below tau.
    # Shading per-window (not a single span) means isolated dips are correctly
    # highlighted even when the streak is not yet consecutive enough to trigger t*.
    checkpoint_freq = epochs[1] - epochs[0] if len(epochs) > 1 else 5

    for epoch_curr, is_below in zip(epochs, below_tau_flags):
        if is_below:
            ax.axvspan(
                epoch_curr - checkpoint_freq, epoch_curr,
                alpha=0.18, color=COLOR_STABLE, zorder=0,
            )

    # --- t* line ---
    if t_star is not None:
        ax.axvline(
            t_star, color=COLOR_TSTAR, linestyle=":", linewidth=2.2,
            label=f"t* = {t_star}  ({metric_label} stabilization)",
            zorder=4,
        )
        # Annotate epoch number at the top of the line using axes-fraction
        # coordinates so the label position is independent of y-scale.
        ax.text(
            t_star, 1.0, f" t*={t_star}",
            transform=ax.get_xaxis_transform(),
            color=COLOR_TSTAR, fontsize=8, va="top", ha="left",
        )
    else:
        ax.text(
            0.97, 0.94, "t* not detected",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#888888", style="italic",
        )

    ax.set_ylabel(f"{metric_label} change  (1 − {metric_label})", fontsize=10)
    ax.set_ylim(bottom=-0.01)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
    ax.set_title(f"{metric_label} temporal change", fontsize=11, loc="left", pad=5)


def plot_stability_curves(
    cka_rows: list[dict],
    drs_rows: list[dict],
    summary: dict,
    tau: float,
    K: int,
    output_path: str,
    logger: logging.Logger,
) -> None:
    """
    Produce the stability-curve figure and save it to output_path.

    If DRS data is present: two-panel figure (CKA top, DRS bottom).
    If DRS data is absent:  one-panel figure (CKA only) with a note.
    """
    has_drs   = len(drs_rows) > 0
    n_subplots = 2 if has_drs else 1

    fig_height = FIGURE_HEIGHT_TWO_PANELS if has_drs else FIGURE_HEIGHT_ONE_PANEL
    fig, axes = plt.subplots(
        n_subplots, 1,
        figsize=(FIGURE_WIDTH_INCHES, fig_height),
        sharex=True,
    )

    # Normalise axes to always be a list, regardless of n_subplots
    if n_subplots == 1:
        axes = [axes]

    t_star_cka  = summary.get("CKA", None)
    t_star_drs  = summary.get("DRS", None)

    cka_epochs      = [r["epoch_curr"]  for r in cka_rows]
    cka_changes     = [r["cka_change"]  for r in cka_rows]
    cka_below_flags = [r["below_tau"]   for r in cka_rows]
    epoch_end       = max(cka_epochs) if cka_epochs else 200

    # CKA panel (always present)
    add_stability_panel(
        ax=axes[0],
        epochs=cka_epochs,
        changes=cka_changes,
        below_tau_flags=cka_below_flags,
        color=COLOR_CKA,
        marker="o",
        metric_label="CKA",
        tau=tau,
        t_star=t_star_cka,
        epoch_end=epoch_end,
    )

    # DRS panel (only if data exists)
    if has_drs:
        drs_epochs      = [r["epoch_curr"]  for r in drs_rows]
        drs_changes     = [r["drs_change"]  for r in drs_rows]
        drs_below_flags = [r["below_tau"]   for r in drs_rows]

        add_stability_panel(
            ax=axes[1],
            epochs=drs_epochs,
            changes=drs_changes,
            below_tau_flags=drs_below_flags,
            color=COLOR_DRS,
            marker="s",
            metric_label="DRS",
            tau=tau,
            t_star=t_star_drs,
            epoch_end=epoch_end,
        )

    axes[-1].set_xlabel("Epoch", fontsize=11)

    tau_label = f"τ = {tau},  K = {K}"
    fig.suptitle(
        f"Representation Stability over Training  ({tau_label})",
        fontsize=13, fontweight="bold",
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
        description="Plot CKA and DRS temporal stability curves with t* marked."
    )
    parser.add_argument(
        "--config", type=str, default="configs/resnet18_cifar10.yaml",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Override results directory from config.",
    )
    parser.add_argument(
        "--tau", type=float, default=None,
        help="Override tau from config (for display only — does not recompute metrics).",
    )
    parser.add_argument(
        "--K", type=int, default=None,
        help="Override K from config (for display only).",
    )
    args = parser.parse_args()

    logger = setup_logging()

    config      = load_config(args.config)
    results_dir = args.results_dir or config["paths"]["results"]
    tau         = args.tau if args.tau is not None else float(config["tau"])
    K           = args.K  if args.K  is not None else int(config["K"])

    logger.info("=" * 60)
    logger.info("plot_stability.py — Representation Stability Curves")
    logger.info("=" * 60)
    logger.info(f"Results dir : {results_dir}")
    logger.info(f"tau = {tau},  K = {K}")

    cka_rows = load_cka_results(results_dir)
    logger.info(f"Loaded {len(cka_rows)} CKA pairs")

    drs_rows = load_drs_results(results_dir)
    if drs_rows:
        logger.info(f"Loaded {len(drs_rows)} DRS pairs")
    else:
        logger.warning(
            "DRS results not found — plotting CKA only. "
            "Run 'python metrics/drs.py' to add DRS panel."
        )

    summary = load_stabilization_summary(results_dir)
    if summary:
        logger.info(f"Stabilization summary: { {k: v for k, v in summary.items()} }")
    else:
        logger.warning(
            "Stabilization summary not found — t* lines will be omitted. "
            "Run 'python metrics/stabilization.py' to add them."
        )

    output_path = os.path.join(results_dir, "plot_stability.png")

    plot_stability_curves(
        cka_rows=cka_rows,
        drs_rows=drs_rows,
        summary=summary,
        tau=tau,
        K=K,
        output_path=output_path,
        logger=logger,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
