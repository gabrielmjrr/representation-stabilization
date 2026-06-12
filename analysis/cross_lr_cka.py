#!/usr/bin/env python3
"""
analysis/cross_lr_cka.py

Cross-learning-rate representation similarity analysis using CKA.

Scientific motivation:
    Frozen-feature probes saturate earlier than CKA-based stabilization.
    This script asks whether different learning rates converge to the same
    representational solution or to genuinely different endpoints.  The output
    distinguishes four scenarios:
        (1) Same endpoint, different speed  -- high final cross-LR CKA,
            off-diagonal ridge in heatmap.
        (2) Different endpoints             -- low final cross-LR CKA despite
            similar test accuracy.
        (3) LR differences < seed noise     -- cross-LR CKA ≈ within-LR
            cross-seed CKA.
        (4) LR genuinely changes solution   -- cross-LR CKA well below the
            within-LR cross-seed baseline.

Computation order:
    PRIMARY:   Epoch×epoch cross-LR CKA heatmaps, one per (seed, LR-pair).
               rows = checkpoints from LR A, cols = checkpoints from LR B.
    DERIVED:   Same-epoch diagonal trajectories, final-final values, and
               max-CKA epoch pairs are all extracted from the heatmaps, not
               computed independently.
    SEPARATE:  Final-epoch all-run 15×15 CKA matrix.  This covers cross-seed
               comparisons that are absent from the seed-matched heatmaps and
               provides the within-LR cross-seed baseline.

Usage:
    python analysis/cross_lr_cka.py \\
        --base-dir /local/data/gme101 \\
        --repo-results-dir thesis_results \\
        --split full_test

    python analysis/cross_lr_cka.py \\
        --base-dir /local/data/gme101 \\
        --epochs 0,10,20,50,100,150,200 \\
        --save-per-seed-plots
"""

import argparse
import logging
import os
import re
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LR_FLOATS = {"005": 0.05, "010": 0.10, "020": 0.20}
FINAL_EPOCH = 200


# ---------------------------------------------------------------------------
# Feature-space linear CKA
# ---------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Feature-space linear CKA between representation matrices X (n,p) and Y (n,q).

    Uses the feature-space formula to avoid constructing n×n Gram matrices,
    which is impractical for n=10 000 (full_test):

        CKA = ||X_c^T Y_c||_F^2 / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F)

    where X_c, Y_c are centered over examples (column means subtracted).
    This equals the Gram-matrix formulation: HSIC(XX^T, YY^T) /
    sqrt(HSIC(XX^T, XX^T) * HSIC(YY^T, YY^T)).

    Returns a float in [0, 1].  Returns 0.0 when either representation has
    near-zero variance (denominator < 1e-10).
    """
    X_c = X - X.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)

    XtY = X_c.T @ Y_c   # (p, q)
    XtX = X_c.T @ X_c   # (p, p)
    YtY = Y_c.T @ Y_c   # (q, q)

    hsic_XY = float(np.sum(XtY ** 2))
    hsic_XX = float(np.sum(XtX ** 2))
    hsic_YY = float(np.sum(YtY ** 2))

    denom = np.sqrt(hsic_XX * hsic_YY)
    if denom < 1e-10:
        return 0.0
    return float(hsic_XY / denom)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-LR CKA analysis on frozen penultimate-layer features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-dir", type=str, default="/local/data/gme101",
        help="Root directory containing per-run subdirectories.",
    )
    parser.add_argument(
        "--repo-results-dir", type=str, default="thesis_results",
        help="Repo-relative directory for aggregate outputs and plots.",
    )
    parser.add_argument(
        "--split", type=str, default="full_test",
        choices=["full_test", "full_train"],
        help=(
            "Feature split to use.  Default is full_test: all runs share the "
            "same test examples in the same order, avoiding cross-seed index "
            "mismatch that would corrupt CKA values."
        ),
    )
    parser.add_argument(
        "--epochs", type=str, default=None,
        help="Comma-separated epoch list for debug/partial runs, e.g. '0,10,50,200'.",
    )
    parser.add_argument(
        "--skip-heatmaps", action="store_true",
        help="Skip epoch×epoch heatmaps; compute only the final all-run matrix.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute outputs even if they already exist on disk.",
    )
    parser.add_argument(
        "--save-per-seed-plots", action="store_true",
        help="Save individual heatmap PNG for every (seed, LR-pair) combination.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("cross_lr_cka")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Run discovery and feature loading
# ---------------------------------------------------------------------------

def discover_runs(base_dir: Path) -> list:
    """
    Find run directories matching final_lr<code>_seed<n> under base_dir.
    Returns list of dicts: run_name, lr_code, lr, seed, path.
    """
    pattern = re.compile(r"^final_lr(\d{3})_seed(\d+)$")
    runs = []
    if not base_dir.is_dir():
        return runs
    for d in sorted(base_dir.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m is None:
            continue
        lr_code = m.group(1)
        seed = int(m.group(2))
        runs.append({
            "run_name": d.name,
            "lr_code": lr_code,
            "lr": LR_FLOATS.get(lr_code, int(lr_code) / 100.0),
            "seed": seed,
            "path": d,
        })
    return runs


def find_feature_epochs(split_dir: Path) -> list:
    """Discover available checkpoint epochs from features_epoch_XXXX.npy filenames."""
    if not split_dir.is_dir():
        return []
    epochs = []
    for f in split_dir.iterdir():
        if f.name.startswith("features_epoch_") and f.name.endswith(".npy"):
            try:
                epochs.append(int(f.name[len("features_epoch_"):-len(".npy")]))
            except ValueError:
                pass
    return sorted(epochs)


def load_run_features(run: dict, split: str, epoch_filter: set,
                      logger: logging.Logger) -> dict:
    """
    Load all available epoch features for one run.
    Returns dict: epoch (int) -> np.ndarray (n, d).
    Skips missing files gracefully.
    """
    split_dir = run["path"] / "activations" / split
    available = find_feature_epochs(split_dir)
    if epoch_filter:
        available = [e for e in available if e in epoch_filter]
    features = {}
    for epoch in available:
        path = split_dir / f"features_epoch_{epoch:04d}.npy"
        if not path.exists():
            logger.warning(f"  Missing feature file: {path}")
            continue
        features[epoch] = np.load(path)
    logger.info(
        f"  {run['run_name']}: loaded {len(features)} epochs "
        f"from {split_dir.name}"
    )
    return features


# ---------------------------------------------------------------------------
# Primary computation: epoch×epoch cross-LR heatmaps
# ---------------------------------------------------------------------------

def compute_cross_lr_heatmap(
    feat_a: dict, feat_b: dict, logger: logging.Logger
) -> tuple:
    """
    Compute the full epoch×epoch cross-LR CKA heatmap.

    This is the primary computation.  All same-epoch diagonal values,
    final-final CKA, and max-CKA summaries are derived from this matrix
    rather than computed independently.

    Returns (epochs_a, epochs_b, matrix) where
        matrix[i, j] = CKA(feat_a[epochs_a[i]], feat_b[epochs_b[j]]).
    """
    epochs_a = sorted(feat_a.keys())
    epochs_b = sorted(feat_b.keys())
    n_a, n_b = len(epochs_a), len(epochs_b)
    matrix = np.full((n_a, n_b), np.nan)

    total = n_a * n_b
    done = 0
    for i, ea in enumerate(epochs_a):
        for j, eb in enumerate(epochs_b):
            matrix[i, j] = linear_cka(feat_a[ea], feat_b[eb])
            done += 1
            if done % 100 == 0 or done == total:
                logger.info(
                    f"    heatmap CKA: {done}/{total}  "
                    f"ep({ea},{eb})={matrix[i,j]:.4f}"
                )
    return epochs_a, epochs_b, matrix


# ---------------------------------------------------------------------------
# Derive summaries from heatmaps
# ---------------------------------------------------------------------------

def derive_same_epoch_rows(
    seed: int, lr_a: float, lr_b: float,
    epochs_a: list, epochs_b: list, matrix: np.ndarray,
) -> list:
    """
    Extract same-epoch diagonal entries from a cross-LR heatmap.
    These are the cells where epoch_a == epoch_b.
    """
    idx_a = {e: i for i, e in enumerate(epochs_a)}
    idx_b = {e: j for j, e in enumerate(epochs_b)}
    rows = []
    for ep in sorted(set(epochs_a) & set(epochs_b)):
        rows.append({
            "seed": seed, "lr_a": lr_a, "lr_b": lr_b,
            "epoch": ep,
            "cka": float(matrix[idx_a[ep], idx_b[ep]]),
            "comparison_type": "seed_matched_cross_lr_same_epoch",
        })
    return rows


def derive_heatmap_summary_row(
    seed: int, lr_a: float, lr_b: float,
    epochs_a: list, epochs_b: list, matrix: np.ndarray,
) -> dict:
    """
    Derive final-final CKA, max CKA, and the epoch pair at max CKA
    from a single cross-LR heatmap.
    """
    idx_a = {e: i for i, e in enumerate(epochs_a)}
    idx_b = {e: j for j, e in enumerate(epochs_b)}

    shared = sorted(set(epochs_a) & set(epochs_b))
    final_ep = shared[-1] if shared else None
    final_final_cka = (
        float(matrix[idx_a[final_ep], idx_b[final_ep]])
        if final_ep is not None else float("nan")
    )

    valid = ~np.isnan(matrix)
    if not valid.any():
        return {
            "seed": seed, "lr_a": lr_a, "lr_b": lr_b,
            "max_cka": float("nan"),
            "epoch_a_at_max": float("nan"), "epoch_b_at_max": float("nan"),
            "epoch_difference_at_max": float("nan"),
            "final_epoch": final_ep, "final_final_cka": final_final_cka,
        }

    i_max, j_max = np.unravel_index(np.nanargmax(matrix), matrix.shape)
    ep_a_max = epochs_a[int(i_max)]
    ep_b_max = epochs_b[int(j_max)]

    return {
        "seed": seed, "lr_a": lr_a, "lr_b": lr_b,
        "max_cka": float(matrix[i_max, j_max]),
        "epoch_a_at_max": ep_a_max,
        "epoch_b_at_max": ep_b_max,
        "epoch_difference_at_max": int(ep_b_max) - int(ep_a_max),
        "final_epoch": final_ep,
        "final_final_cka": final_final_cka,
    }


def build_mean_heatmaps(all_heatmaps: dict, lr_pairs: list, logger: logging.Logger) -> dict:
    """
    Average per-seed heatmaps for each LR pair, aligned by epoch label.
    Returns dict: (lr_a, lr_b) -> (epochs_a, epochs_b, mean_matrix).
    """
    mean_heatmaps = {}
    for lr_a, lr_b in lr_pairs:
        # Collect seed heatmaps for this LR pair
        seed_entries = [
            (ep_a, ep_b, mat)
            for (a, b, _s), (ep_a, ep_b, mat) in all_heatmaps.items()
            if a == lr_a and b == lr_b
        ]
        if not seed_entries:
            continue

        # Align on the intersection of available epochs across seeds
        common_a = set(seed_entries[0][0])
        common_b = set(seed_entries[0][1])
        for ep_a, ep_b, _ in seed_entries[1:]:
            common_a &= set(ep_a)
            common_b &= set(ep_b)

        epochs_a = sorted(common_a)
        epochs_b = sorted(common_b)

        stack = []
        for ep_a, ep_b, mat in seed_entries:
            ia = [ep_a.index(e) for e in epochs_a]
            ib = [ep_b.index(e) for e in epochs_b]
            stack.append(mat[np.ix_(ia, ib)])

        mean_mat = np.nanmean(stack, axis=0)
        mean_heatmaps[(lr_a, lr_b)] = (epochs_a, epochs_b, mean_mat)
        logger.info(
            f"  Mean heatmap LR={lr_a:.2f} vs LR={lr_b:.2f}: "
            f"{len(stack)} seed(s), shape {mean_mat.shape}"
        )
    return mean_heatmaps


# ---------------------------------------------------------------------------
# Final-epoch all-run CKA matrix (computed separately from heatmaps)
# ---------------------------------------------------------------------------

def compute_final_epoch_matrix(
    runs: list, split: str, final_epoch: int, logger: logging.Logger
) -> tuple:
    """
    Compute pairwise final-epoch CKA across all available runs.

    This matrix is computed separately from the seed-matched heatmaps because
    it includes cross-seed comparisons (e.g. LR=0.05 seed=0 vs LR=0.10 seed=1)
    that are absent from the seed-matched cross-LR heatmaps.  These cross-seed
    off-diagonal entries provide the within-LR cross-seed baseline.

    Returns (valid_runs, matrix) where valid_runs is the subset of runs for
    which the final-epoch feature file was found.
    """
    feat = {}
    valid_runs = []
    for run in runs:
        path = run["path"] / "activations" / split / f"features_epoch_{final_epoch:04d}.npy"
        if not path.exists():
            logger.warning(f"  Skipping {run['run_name']}: missing {path.name}")
            continue
        feat[run["run_name"]] = np.load(path)
        valid_runs.append(run)
        logger.info(f"  Loaded final epoch for {run['run_name']}")

    n = len(valid_runs)
    matrix = np.zeros((n, n))
    total = n * (n + 1) // 2
    done = 0
    for i in range(n):
        for j in range(i, n):
            name_i = valid_runs[i]["run_name"]
            name_j = valid_runs[j]["run_name"]
            cka_val = 1.0 if i == j else linear_cka(feat[name_i], feat[name_j])
            matrix[i, j] = cka_val
            matrix[j, i] = cka_val
            done += 1
            if done % 20 == 0 or done == total:
                logger.info(
                    f"  final matrix: {done}/{total}  "
                    f"{name_i} vs {name_j}  CKA={cka_val:.4f}"
                )
    return valid_runs, matrix


# ---------------------------------------------------------------------------
# Summary aggregations
# ---------------------------------------------------------------------------

def build_within_lr_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Within-LR cross-seed final CKA baseline from the all-run long format.
    Includes only same-lr, different-seed pairs (no self-comparisons).
    """
    mask = (long_df["same_lr"] == True) & (long_df["same_seed"] == False)
    sub = long_df[mask]
    if sub.empty:
        return pd.DataFrame(
            columns=["lr", "mean_within_lr_cross_seed_cka",
                     "std_within_lr_cross_seed_cka", "n_pairs"]
        )
    return (
        sub.groupby("lr_a")["cka"]
        .agg(
            mean_within_lr_cross_seed_cka="mean",
            std_within_lr_cross_seed_cka="std",
            n_pairs="count",
        )
        .reset_index()
        .rename(columns={"lr_a": "lr"})
    )


def build_cross_lr_final_summary(heatmap_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate seed-matched final-final CKA by LR pair."""
    if heatmap_summary_df.empty:
        return pd.DataFrame()
    return (
        heatmap_summary_df.groupby(["lr_a", "lr_b"])["final_final_cka"]
        .agg(
            mean_final_cka="mean",
            std_final_cka="std",
            n="count",
            min_final_cka="min",
            max_final_cka="max",
        )
        .reset_index()
    )


def build_combined_summary(within_df: pd.DataFrame, cross_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compact comparison table: within-LR seed variation vs cross-LR differences.
    Allows direct reading of whether LR effects exceed seed-level noise.
    """
    rows = []
    for _, r in within_df.sort_values("lr").iterrows():
        rows.append({
            "comparison": f"within LR={r['lr']:.2f} across seeds",
            "comparison_type": "within_lr_cross_seed",
            "mean_final_cka": r["mean_within_lr_cross_seed_cka"],
            "std_final_cka": r["std_within_lr_cross_seed_cka"],
            "n": int(r["n_pairs"]),
        })
    for _, r in cross_df.sort_values(["lr_a", "lr_b"]).iterrows():
        rows.append({
            "comparison": f"LR={r['lr_a']:.2f} vs LR={r['lr_b']:.2f}, seed-matched",
            "comparison_type": "cross_lr_seed_matched",
            "mean_final_cka": r["mean_final_cka"],
            "std_final_cka": r["std_final_cka"],
            "n": int(r["n"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def save_heatmap_csv(
    epochs_a: list, epochs_b: list, matrix: np.ndarray, path: Path
) -> None:
    """Save an epoch×epoch heatmap as wide CSV with epoch labels on both axes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(matrix, index=epochs_a, columns=epochs_b)
    df.index.name = "epoch_a"
    df.to_csv(path, float_format="%.6f")


def save_df(df: pd.DataFrame, path: Path, logger: logging.Logger, label: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"{label}: saved {len(df)} rows  ->  {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_LR_PAIR_COLORS = {
    (0.05, 0.10): "#1f77b4",
    (0.05, 0.20): "#ff7f0e",
    (0.10, 0.20): "#2ca02c",
}


def plot_same_epoch_trajectory(
    summary_df: pd.DataFrame, plot_dir: Path, logger: logging.Logger
) -> None:
    """Cross-LR CKA over epochs: mean ± 1 std across seeds for each LR pair."""
    if summary_df.empty:
        logger.warning("same_epoch_summary empty; skipping trajectory plot.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for (lr_a, lr_b), sub in summary_df.groupby(["lr_a", "lr_b"]):
        color = _LR_PAIR_COLORS.get((lr_a, lr_b))
        label = f"LR={lr_a:.2f} vs LR={lr_b:.2f}"
        std = sub["std_cka"].fillna(0.0)
        ax.plot(sub["epoch"], sub["mean_cka"], color=color, linewidth=1.5, label=label)
        ax.fill_between(
            sub["epoch"],
            sub["mean_cka"] - std,
            sub["mean_cka"] + std,
            alpha=0.15, color=color,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CKA")
    ax.set_ylim(0, 1)
    ax.set_title("Cross-LR representation similarity over training (same epoch)")
    ax.legend()
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    out = plot_dir / "cross_lr_cka_same_epoch_trajectory.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")


def plot_final_epoch_matrix(
    matrix_df: pd.DataFrame, valid_runs: list,
    plot_dir: Path, logger: logging.Logger,
) -> None:
    """Heatmap of final-epoch pairwise CKA across all runs, grouped by LR."""
    if matrix_df.empty:
        logger.warning("final_epoch_matrix empty; skipping heatmap.")
        return
    n = len(valid_runs)
    run_labels = [f"LR={r['lr']:.2f}\ns{r['seed']}" for r in valid_runs]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.75), max(7, n * 0.7)))
    im = ax.imshow(
        matrix_df.values.astype(float),
        vmin=0, vmax=1, cmap="viridis", aspect="auto",
    )
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(run_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(run_labels, fontsize=7)

    # Thin white separators between LR groups
    lrs = [r["lr"] for r in valid_runs]
    for i in range(1, n):
        if lrs[i] != lrs[i - 1]:
            ax.axhline(i - 0.5, color="white", linewidth=1.5)
            ax.axvline(i - 0.5, color="white", linewidth=1.5)

    plt.colorbar(im, ax=ax, label="CKA", fraction=0.046, pad=0.04)
    ax.set_title(f"Final-epoch (ep {FINAL_EPOCH}) pairwise CKA across all runs")
    fig.tight_layout()
    out = plot_dir / "final_epoch_all_runs_cka_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")


def plot_final_similarity_bar(
    combined_df: pd.DataFrame, plot_dir: Path, logger: logging.Logger
) -> None:
    """Bar chart: within-LR seed variation vs cross-LR final CKA."""
    if combined_df.empty:
        logger.warning("combined_summary empty; skipping bar plot.")
        return
    x = np.arange(len(combined_df))
    colors = [
        "#1f77b4" if r == "within_lr_cross_seed" else "#d62728"
        for r in combined_df["comparison_type"]
    ]
    fig, ax = plt.subplots(figsize=(max(9, len(combined_df) * 1.3), 5))
    ax.bar(x, combined_df["mean_final_cka"], color=colors, alpha=0.75, width=0.6)
    std = combined_df["std_final_cka"].fillna(0.0)
    ax.errorbar(
        x, combined_df["mean_final_cka"], yerr=std,
        fmt="none", color="black", capsize=4, linewidth=1,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(combined_df["comparison"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"Final CKA (epoch {FINAL_EPOCH})")
    ax.set_ylim(0, 1)
    ax.set_title("Final-epoch representation similarity: within-LR vs cross-LR")
    ax.grid(True, axis="y", linewidth=0.4)
    fig.tight_layout()
    out = plot_dir / "final_similarity_summary_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")


def _heatmap_epoch_ticks(epochs: list, max_ticks: int = 10) -> tuple:
    step = max(1, len(epochs) // max_ticks)
    positions = list(range(0, len(epochs), step))
    labels = [str(epochs[i]) for i in positions]
    return positions, labels


def plot_mean_heatmap(
    lr_a: float, lr_b: float,
    epochs_a: list, epochs_b: list, mean_mat: np.ndarray,
    plot_dir: Path, logger: logging.Logger,
) -> None:
    """Mean epoch×epoch cross-LR CKA heatmap averaged across seeds."""
    lr_a_code = f"{int(round(lr_a * 100)):03d}"
    lr_b_code = f"{int(round(lr_b * 100)):03d}"

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mean_mat, vmin=0, vmax=1, cmap="viridis", aspect="auto", origin="upper")

    ypos, ylabels = _heatmap_epoch_ticks(epochs_a)
    xpos, xlabels = _heatmap_epoch_ticks(epochs_b)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel(f"Epoch  (LR={lr_a:.2f})")
    ax.set_xlabel(f"Epoch  (LR={lr_b:.2f})")
    ax.set_title(
        f"Mean cross-LR CKA: LR={lr_a:.2f} vs LR={lr_b:.2f} "
        f"(averaged across seeds)"
    )
    plt.colorbar(im, ax=ax, label="CKA", fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = plot_dir / f"mean_heatmap_lr{lr_a_code}_vs_lr{lr_b_code}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")


def plot_per_seed_heatmap(
    lr_a: float, lr_b: float, seed: int,
    epochs_a: list, epochs_b: list, matrix: np.ndarray,
    plot_dir: Path, logger: logging.Logger,
) -> None:
    """Per-seed cross-LR epoch×epoch heatmap (optional)."""
    lr_a_code = f"{int(round(lr_a * 100)):03d}"
    lr_b_code = f"{int(round(lr_b * 100)):03d}"

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto", origin="upper")
    ypos, ylabels = _heatmap_epoch_ticks(epochs_a, max_ticks=8)
    xpos, xlabels = _heatmap_epoch_ticks(epochs_b, max_ticks=8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel(f"Epoch  (LR={lr_a:.2f})")
    ax.set_xlabel(f"Epoch  (LR={lr_b:.2f})")
    ax.set_title(f"Cross-LR CKA: LR={lr_a:.2f} vs LR={lr_b:.2f}, seed={seed}")
    plt.colorbar(im, ax=ax, label="CKA", fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = plot_dir / f"heatmap_lr{lr_a_code}_vs_lr{lr_b_code}_seed{seed}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    base_dir = Path(args.base_dir)
    out_root = Path(args.repo_results_dir) / "_aggregate" / "cross_lr_cka"
    plot_dir = Path(args.repo_results_dir) / "_aggregate" / "plots" / "cross_lr_cka"
    heatmap_csv_dir = out_root / "heatmaps_csv"
    log_path = Path(args.repo_results_dir) / "_aggregate" / "logs" / "cross_lr_cka.log"

    for d in (out_root, heatmap_csv_dir, plot_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_path)
    logger.info("=" * 60)
    logger.info("cross_lr_cka.py  start")
    logger.info(f"  base_dir         = {base_dir}")
    logger.info(f"  split            = {args.split}")
    logger.info(f"  skip_heatmaps    = {args.skip_heatmaps}")
    logger.info(f"  overwrite        = {args.overwrite}")
    logger.info(f"  save_per_seed    = {args.save_per_seed_plots}")
    logger.info("=" * 60)

    epoch_filter: set = set()
    if args.epochs:
        epoch_filter = {int(e.strip()) for e in args.epochs.split(",")}
        logger.info(f"Epoch filter: {sorted(epoch_filter)}")

    # --- Discover runs ---
    all_runs = discover_runs(base_dir)
    if not all_runs:
        logger.error(f"No matching run directories found under {base_dir}.")
        sys.exit(1)
    logger.info(f"Discovered {len(all_runs)} run(s).")

    run_index = {(r["lr"], r["seed"]): r for r in all_runs}
    available_lrs = sorted({r["lr"] for r in all_runs})
    available_seeds = sorted({r["seed"] for r in all_runs})
    lr_pairs = list(combinations(available_lrs, 2))

    logger.info(f"Available LRs:   {available_lrs}")
    logger.info(f"Available seeds: {available_seeds}")
    logger.info(f"LR pairs:        {lr_pairs}")

    # ===================================================================
    # PART 1: Epoch×epoch cross-LR heatmaps  (primary computation)
    #
    # Same-epoch diagonal trajectories and final-final values are derived
    # from these heatmaps, not computed separately.
    # ===================================================================

    all_same_epoch_rows = []
    all_heatmap_summary_rows = []
    # key: (lr_a, lr_b, seed) -> (epochs_a, epochs_b, matrix)
    all_heatmaps = {}
    same_epoch_summary = pd.DataFrame()
    cross_lr_final_summary = pd.DataFrame()
    mean_heatmaps = {}

    if not args.skip_heatmaps:
        for seed in available_seeds:
            logger.info(f"--- Seed {seed} ---")

            seed_lrs = [lr for lr in available_lrs if (lr, seed) in run_index]
            if len(seed_lrs) < 2:
                logger.warning(f"  Seed {seed}: fewer than 2 LRs available, skipping.")
                continue

            # Load all epoch features for each LR at this seed.
            # Kept in seed_feat until all three LR pairs for this seed are done,
            # then deleted to free memory before moving to the next seed.
            seed_feat = {}
            for lr in seed_lrs:
                run = run_index[(lr, seed)]
                logger.info(f"  Loading: {run['run_name']}")
                seed_feat[lr] = load_run_features(run, args.split, epoch_filter, logger)

            for lr_a, lr_b in lr_pairs:
                if lr_a not in seed_feat or lr_b not in seed_feat:
                    logger.warning(
                        f"  Skipping LR={lr_a:.2f} vs LR={lr_b:.2f} "
                        f"seed={seed}: run missing."
                    )
                    continue
                if not seed_feat[lr_a] or not seed_feat[lr_b]:
                    logger.warning(
                        f"  Skipping LR={lr_a:.2f} vs LR={lr_b:.2f} "
                        f"seed={seed}: no features loaded."
                    )
                    continue

                lr_a_code = f"{int(round(lr_a * 100)):03d}"
                lr_b_code = f"{int(round(lr_b * 100)):03d}"
                heatmap_path = (
                    heatmap_csv_dir
                    / f"heatmap_lr{lr_a_code}_vs_lr{lr_b_code}_seed{seed}.csv"
                )

                if not args.overwrite and heatmap_path.exists():
                    logger.info(f"  Loading cached heatmap: {heatmap_path.name}")
                    df = pd.read_csv(heatmap_path, index_col="epoch_a")
                    epochs_a = [int(x) for x in df.index.tolist()]
                    epochs_b = [int(x) for x in df.columns.tolist()]
                    matrix = df.values.astype(float)
                else:
                    logger.info(
                        f"  Computing heatmap: "
                        f"LR={lr_a:.2f} vs LR={lr_b:.2f}  seed={seed}"
                    )
                    epochs_a, epochs_b, matrix = compute_cross_lr_heatmap(
                        seed_feat[lr_a], seed_feat[lr_b], logger
                    )
                    save_heatmap_csv(epochs_a, epochs_b, matrix, heatmap_path)
                    logger.info(f"  Saved heatmap: {heatmap_path.name}")

                all_heatmaps[(lr_a, lr_b, seed)] = (epochs_a, epochs_b, matrix)

                # Derived: same-epoch diagonal and per-heatmap summary
                all_same_epoch_rows.extend(
                    derive_same_epoch_rows(seed, lr_a, lr_b, epochs_a, epochs_b, matrix)
                )
                all_heatmap_summary_rows.append(
                    derive_heatmap_summary_row(seed, lr_a, lr_b, epochs_a, epochs_b, matrix)
                )

                if args.save_per_seed_plots:
                    plot_per_seed_heatmap(
                        lr_a, lr_b, seed, epochs_a, epochs_b, matrix, plot_dir, logger
                    )

            del seed_feat  # release memory before next seed

        # --- Save derived CSVs ---
        same_epoch_df = pd.DataFrame(all_same_epoch_rows)
        save_df(
            same_epoch_df,
            out_root / "cross_lr_cka_same_epoch_long.csv",
            logger, "[same_epoch_long]",
        )

        if not same_epoch_df.empty:
            same_epoch_summary = (
                same_epoch_df.groupby(["lr_a", "lr_b", "epoch"])["cka"]
                .agg(mean_cka="mean", std_cka="std", n="count")
                .reset_index()
            )
            save_df(
                same_epoch_summary,
                out_root / "cross_lr_cka_same_epoch_summary.csv",
                logger, "[same_epoch_summary]",
            )

        heatmap_summary_df = pd.DataFrame(all_heatmap_summary_rows)
        save_df(
            heatmap_summary_df,
            out_root / "cross_lr_cka_heatmap_summary.csv",
            logger, "[heatmap_summary]",
        )

        if not heatmap_summary_df.empty:
            cross_lr_final_summary = build_cross_lr_final_summary(heatmap_summary_df)
            save_df(
                cross_lr_final_summary,
                out_root / "cross_lr_cka_final_summary.csv",
                logger, "[cross_lr_final_summary]",
            )

        # Mean heatmaps across seeds
        logger.info("Building mean heatmaps across seeds...")
        mean_heatmaps = build_mean_heatmaps(all_heatmaps, lr_pairs, logger)
        for (lr_a, lr_b), (ep_a, ep_b, mean_mat) in mean_heatmaps.items():
            lr_a_code = f"{int(round(lr_a * 100)):03d}"
            lr_b_code = f"{int(round(lr_b * 100)):03d}"
            mean_path = heatmap_csv_dir / f"heatmap_lr{lr_a_code}_vs_lr{lr_b_code}_mean.csv"
            save_heatmap_csv(ep_a, ep_b, mean_mat, mean_path)
            logger.info(f"Saved mean heatmap: {mean_path.name}")

    else:
        logger.info("Heatmap computation skipped (--skip-heatmaps).")

    # ===================================================================
    # PART 2: Final-epoch all-run CKA matrix
    #
    # Computed separately from the seed-matched heatmaps because it
    # includes cross-seed comparisons (e.g. LR=0.05 seed=0 vs LR=0.10
    # seed=1) that are absent from the seed-matched objects above.
    # These cross-seed off-diagonals provide the within-LR baseline.
    # ===================================================================

    final_matrix_path = out_root / "final_epoch_cka_matrix.csv"
    final_long_path = out_root / "final_epoch_cka_long.csv"

    valid_runs_final = []
    final_matrix_df = pd.DataFrame()
    final_long_df = pd.DataFrame()

    if not args.overwrite and final_matrix_path.exists() and final_long_path.exists():
        logger.info("Loading cached final-epoch matrix files.")
        final_matrix_df = pd.read_csv(final_matrix_path, index_col=0)
        final_long_df = pd.read_csv(final_long_path)
        # Restore run ordering from matrix index
        name_to_run = {r["run_name"]: r for r in all_runs}
        valid_runs_final = [
            name_to_run[n]
            for n in final_matrix_df.index.tolist()
            if n in name_to_run
        ]
    else:
        logger.info("Computing final-epoch all-run CKA matrix...")
        sorted_runs = sorted(all_runs, key=lambda r: (r["lr"], r["seed"]))
        valid_runs_final, matrix = compute_final_epoch_matrix(
            sorted_runs, args.split, FINAL_EPOCH, logger
        )

        if valid_runs_final:
            run_names = [r["run_name"] for r in valid_runs_final]
            final_matrix_df = pd.DataFrame(matrix, index=run_names, columns=run_names)
            final_matrix_df.index.name = "run_name"
            final_matrix_df.to_csv(final_matrix_path, float_format="%.6f")
            logger.info(f"Saved: {final_matrix_path}")

            n = len(valid_runs_final)
            long_rows = []
            for i in range(n):
                for j in range(i, n):
                    ri, rj = valid_runs_final[i], valid_runs_final[j]
                    long_rows.append({
                        "run_a": ri["run_name"], "lr_a": ri["lr"], "seed_a": ri["seed"],
                        "run_b": rj["run_name"], "lr_b": rj["lr"], "seed_b": rj["seed"],
                        "cka": float(matrix[i, j]),
                        "same_lr": ri["lr"] == rj["lr"],
                        "same_seed": ri["seed"] == rj["seed"],
                    })
            final_long_df = pd.DataFrame(long_rows)
            save_df(final_long_df, final_long_path, logger, "[final_long]")

    # Within-LR cross-seed baseline
    within_lr_summary = pd.DataFrame()
    if not final_long_df.empty:
        within_lr_summary = build_within_lr_summary(final_long_df)
        save_df(
            within_lr_summary,
            out_root / "within_lr_cross_seed_final_summary.csv",
            logger, "[within_lr_summary]",
        )

    # Combined summary
    combined_df = pd.DataFrame()
    if not within_lr_summary.empty and not cross_lr_final_summary.empty:
        combined_df = build_combined_summary(within_lr_summary, cross_lr_final_summary)
        save_df(
            combined_df,
            out_root / "combined_final_similarity_summary.csv",
            logger, "[combined_summary]",
        )

    # ===================================================================
    # PART 3: Plots
    # ===================================================================

    logger.info("Creating plots...")

    if not same_epoch_summary.empty:
        plot_same_epoch_trajectory(same_epoch_summary, plot_dir, logger)

    if not final_matrix_df.empty and valid_runs_final:
        plot_final_epoch_matrix(final_matrix_df, valid_runs_final, plot_dir, logger)

    if not combined_df.empty:
        plot_final_similarity_bar(combined_df, plot_dir, logger)

    for (lr_a, lr_b), (ep_a, ep_b, mean_mat) in mean_heatmaps.items():
        plot_mean_heatmap(lr_a, lr_b, ep_a, ep_b, mean_mat, plot_dir, logger)

    logger.info("cross_lr_cka.py  done.")


if __name__ == "__main__":
    main()
