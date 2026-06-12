#!/usr/bin/env python3
"""
spectral_components.py

Deng-et-al.-style spectral component analysis on frozen penultimate-layer features.

Scientific motivation:
    Dominant spectral components of the representation may become readout-accessible
    early in training, while residual lower-energy components continue evolving.
    This analysis tests whether the gap between probe accuracy saturation and CKA-based
    representational stabilization can be explained by the structure of dominant
    singular directions.

For each run and epoch:
    1. Load train/test features (penultimate layer, ResNet-18).
    2. Center features using the train mean (no leakage to test).
    3. Compute economy SVD on centered train features.
    4. Select k dominant singular directions via threshold on cumulative singular mass
       (default: sum_{i<=k} sigma_i / sum sigma_j >= 0.80).
    5. Project both splits to principal-component coordinates (no reconstruction).
    6. Train logistic regression probes on:
           full     -- all d principal-component directions
           main     -- top-k dominant singular directions
           residual -- remaining d-k lower-energy directions
    7. Record spectral metrics and probe readout accuracy.

Usage:
    python analysis/spectral_components.py \\
        --base-dir /local/data/gme101 \\
        --repo-results-dir thesis_results \\
        --runs "final_lr*_seed*" \\
        --threshold-mode singular_mass \\
        --threshold-value 0.80 \\
        --probe logistic_regression
"""

import argparse
import csv
import glob
import logging
import os
import re
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_RUNS = [
    "final_lr005_seed0", "final_lr005_seed1", "final_lr005_seed2",
    "final_lr005_seed3", "final_lr005_seed42",
    "final_lr010_seed0", "final_lr010_seed1", "final_lr010_seed2",
    "final_lr010_seed3", "final_lr010_seed42",
    "final_lr020_seed0", "final_lr020_seed1", "final_lr020_seed2",
    "final_lr020_seed3", "final_lr020_seed42",
]

SUMMARY_FIELDS = [
    "run_name", "lr", "seed", "epoch",
    "threshold_mode", "threshold_value",
    "k_main", "d_total", "k_fraction",
    "singular_sum_total", "singular_energy_total",
    "main_singular_mass", "main_energy_mass",
    "residual_singular_mass", "residual_energy_mass",
    "effective_rank_singular", "effective_rank_energy",
]

PROBE_FIELDS = [
    "run_name", "lr", "seed", "epoch",
    "component", "probe",
    "train_acc", "test_acc",
    "k_main", "threshold_mode", "threshold_value",
]

SV_FIELDS = [
    "run_name", "lr", "seed", "epoch",
    "component_index", "singular_value", "singular_energy",
    "normalized_singular_mass", "normalized_energy_mass",
]

# Epochs at which singular spectrum snapshots are produced (optional plot).
SNAPSHOT_EPOCHS = {0, 10, 20, 50, 100, 150, 200}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spectral component analysis on frozen penultimate-layer features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-dir", type=str, default="/local/data/gme101",
        help="Root directory containing per-run subdirectories with activations/.",
    )
    parser.add_argument(
        "--repo-results-dir", type=str, default="thesis_results",
        help="Repo-relative directory for aggregate results and plots.",
    )
    parser.add_argument(
        "--runs", type=str, default="final_lr*_seed*",
        help="Glob pattern for run subdirectory names under --base-dir.",
    )
    parser.add_argument(
        "--threshold-mode", type=str, default="singular_mass",
        choices=["singular_mass"],
        help=(
            "Method for selecting k dominant singular directions. "
            "'singular_mass': smallest k s.t. sum(S[:k])/sum(S) >= threshold-value."
        ),
    )
    parser.add_argument(
        "--threshold-value", type=float, default=0.80,
        help="Cumulative singular mass threshold for dominant-component selection.",
    )
    parser.add_argument(
        "--probe", type=str, default="logistic_regression",
        choices=["logistic_regression"],
        help="Probe type for readout-accessibility evaluation.",
    )
    parser.add_argument(
        "--epochs", type=int, nargs="*", default=None,
        help="Restrict analysis to specific epochs (useful for debugging).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute per-run outputs even if they already exist.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for probe fitting.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("spectral_components")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


# ---------------------------------------------------------------------------
# Run discovery and name parsing
# ---------------------------------------------------------------------------

def discover_runs(base_dir: str, pattern: str) -> list:
    """Return sorted list of run directory paths matching glob pattern under base_dir."""
    matched = glob.glob(os.path.join(base_dir, pattern))
    return sorted(d for d in matched if os.path.isdir(d))


def parse_run_name(run_name: str):
    """
    Extract (lr_float, seed_int) from names like 'final_lr010_seed42'.

    Returns None if the name does not match the expected pattern.
    The three-digit lr suffix encodes the learning rate * 100
    (e.g. '005' -> 0.05, '010' -> 0.10, '020' -> 0.20).
    """
    m = re.match(r".*final_lr(\d+)_seed(\d+)$", run_name)
    if m is None:
        return None
    lr = int(m.group(1)) / 100.0
    seed = int(m.group(2))
    return lr, seed


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def find_feature_epochs(train_dir: str) -> list:
    """Discover available checkpoint epochs from features_epoch_XXXX.npy filenames."""
    if not os.path.isdir(train_dir):
        return []
    epochs = []
    for fname in os.listdir(train_dir):
        if fname.startswith("features_epoch_") and fname.endswith(".npy"):
            epoch_str = fname[len("features_epoch_"):-len(".npy")]
            try:
                epochs.append(int(epoch_str))
            except ValueError:
                pass
    return sorted(epochs)


def load_epoch_features(train_dir: str, test_dir: str, epoch: int):
    """Load train and test feature arrays for a single checkpoint epoch."""
    fname = f"features_epoch_{epoch:04d}.npy"
    train_path = os.path.join(train_dir, fname)
    test_path = os.path.join(test_dir, fname)
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train features missing: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test features missing: {test_path}")
    return np.load(train_path), np.load(test_path)


# ---------------------------------------------------------------------------
# Spectral utilities
# ---------------------------------------------------------------------------

def select_k_threshold(S: np.ndarray, mode: str, value: float) -> int:
    """
    Select the number of dominant singular directions k.

    For 'singular_mass': find the smallest k such that
        sum(S[:k]) / sum(S) >= value.

    Parameters
    ----------
    S     : singular values sorted descending, shape (d,)
    mode  : currently only 'singular_mass'
    value : threshold (e.g. 0.80)

    Returns
    -------
    k : 1 <= k <= d
    """
    if mode == "singular_mass":
        cumulative = np.cumsum(S) / S.sum()
        # searchsorted returns leftmost index where cumulative >= value
        idx = int(np.searchsorted(cumulative, value, side="left"))
        k = idx + 1  # convert 0-based index to count
        return min(k, len(S))
    raise ValueError(f"Unknown threshold mode: {mode!r}")


def _safe_entropy(p: np.ndarray) -> float:
    """Shannon entropy of p, skipping zero entries to avoid log(0)."""
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def effective_rank_singular(S: np.ndarray) -> float:
    """
    Effective rank based on the singular-value mass distribution:
        p_i = sigma_i / sum(sigma)
        eff_rank = exp(H(p))
    """
    p = S / S.sum()
    return float(np.exp(_safe_entropy(p)))


def effective_rank_energy(S: np.ndarray) -> float:
    """
    Effective rank based on the squared singular-value (energy) distribution:
        q_i = sigma_i^2 / sum(sigma^2)
        eff_rank = exp(H(q))
    """
    q = S ** 2 / (S ** 2).sum()
    return float(np.exp(_safe_entropy(q)))


def compute_spectral_metrics(
    S: np.ndarray,
    k: int,
    threshold_mode: str,
    threshold_value: float,
) -> dict:
    """
    Scalar spectral summary for a single epoch given sorted singular values S
    and the dominant-component split index k.
    """
    S_main = S[:k]

    singular_sum = float(S.sum())
    singular_energy = float((S ** 2).sum())

    main_singular_mass = float(S_main.sum() / singular_sum) if singular_sum > 0 else 0.0
    main_energy_mass = (
        float((S_main ** 2).sum() / singular_energy) if singular_energy > 0 else 0.0
    )

    return {
        "threshold_mode": threshold_mode,
        "threshold_value": threshold_value,
        "k_main": k,
        "d_total": len(S),
        "k_fraction": float(k / len(S)),
        "singular_sum_total": singular_sum,
        "singular_energy_total": singular_energy,
        "main_singular_mass": main_singular_mass,
        "main_energy_mass": main_energy_mass,
        "residual_singular_mass": float(1.0 - main_singular_mass),
        "residual_energy_mass": float(1.0 - main_energy_mass),
        "effective_rank_singular": effective_rank_singular(S),
        "effective_rank_energy": effective_rank_energy(S),
    }


def build_sv_rows(
    run_name: str,
    lr: float,
    seed: int,
    epoch: int,
    S: np.ndarray,
) -> list:
    """Build per-singular-value rows for the long-format singular values file."""
    total_mass = float(S.sum())
    total_energy = float((S ** 2).sum())
    rows = []
    for i, sv in enumerate(S):
        rows.append({
            "run_name": run_name,
            "lr": lr,
            "seed": seed,
            "epoch": epoch,
            "component_index": i,
            "singular_value": float(sv),
            "singular_energy": float(sv ** 2),
            "normalized_singular_mass": float(sv / total_mass) if total_mass > 0 else 0.0,
            "normalized_energy_mass": (
                float(sv ** 2 / total_energy) if total_energy > 0 else 0.0
            ),
        })
    return rows


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def build_probe(seed: int):
    """
    Logistic regression probe with StandardScaler pre-processing.

    Matches the spirit of Deng et al.'s retrained softmax classifier:
    a linear readout trained on frozen representations to assess
    readout-accessible information in each spectral component.
    """
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            multi_class="auto",
            n_jobs=-1,
            random_state=seed,
        ),
    )


def fit_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
):
    """
    Fit a logistic regression probe and evaluate readout accessibility.

    Returns (train_acc, test_acc, fit_seconds).
    """
    clf = build_probe(seed)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0
    train_acc = float((clf.predict(X_train) == y_train).mean())
    test_acc = float((clf.predict(X_test) == y_test).mean())
    return train_acc, test_acc, fit_seconds


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------

def process_run(
    run_dir: str,
    run_name: str,
    lr: float,
    seed: int,
    args: argparse.Namespace,
    logger: logging.Logger,
):
    """
    Compute spectral decomposition and probe readout accuracy at every available
    epoch for one run, then write per-run CSVs.

    Returns (summary_rows, probe_rows, sv_rows), or (None, None, None) if the
    run should be skipped.
    """
    train_dir = os.path.join(run_dir, "activations", "full_train")
    test_dir = os.path.join(run_dir, "activations", "full_test")
    results_dir = os.path.join(run_dir, "results")

    summary_path = os.path.join(results_dir, "spectral_component_summary.csv")
    probe_path = os.path.join(results_dir, "spectral_probe_results.csv")
    sv_path = os.path.join(results_dir, "spectral_singular_values.csv")

    if not args.overwrite and all(
        os.path.exists(p) for p in (summary_path, probe_path, sv_path)
    ):
        logger.info(
            f"[{run_name}] Per-run outputs exist; loading cached results "
            f"(use --overwrite to recompute)."
        )
        try:
            return (
                pd.read_csv(summary_path).to_dict("records"),
                pd.read_csv(probe_path).to_dict("records"),
                pd.read_csv(sv_path).to_dict("records"),
            )
        except Exception as exc:
            logger.warning(f"[{run_name}] Could not reload cached files ({exc}); recomputing.")

    # --- Labels ---
    train_labels_path = os.path.join(train_dir, "labels.npy")
    test_labels_path = os.path.join(test_dir, "labels.npy")
    if not os.path.exists(train_labels_path):
        logger.warning(f"[{run_name}] Train labels missing at {train_labels_path}, skipping.")
        return None, None, None
    if not os.path.exists(test_labels_path):
        logger.warning(f"[{run_name}] Test labels missing at {test_labels_path}, skipping.")
        return None, None, None

    y_train = np.load(train_labels_path)
    y_test = np.load(test_labels_path)

    # --- Epoch discovery ---
    available_epochs = find_feature_epochs(train_dir)
    if not available_epochs:
        logger.warning(
            f"[{run_name}] No feature files found in {train_dir}, skipping."
        )
        return None, None, None

    if args.epochs is not None:
        epochs = sorted(set(available_epochs) & set(args.epochs))
        logger.info(
            f"[{run_name}] Restricted to {len(epochs)}/{len(available_epochs)} epochs."
        )
    else:
        epochs = available_epochs

    logger.info(f"[{run_name}] Processing {len(epochs)} epochs  (lr={lr}, seed={seed}).")
    os.makedirs(results_dir, exist_ok=True)

    summary_rows = []
    probe_rows = []
    sv_rows = []

    for epoch in epochs:
        logger.info(f"  [{run_name}] epoch {epoch:3d}")

        try:
            X_train, X_test = load_epoch_features(train_dir, test_dir, epoch)
        except FileNotFoundError as exc:
            logger.warning(f"  [{run_name}] epoch {epoch}: {exc}  -- skipping.")
            continue

        # Center using train mean (no leakage to test split).
        mu = X_train.mean(axis=0)
        X_train_c = X_train - mu
        X_test_c = X_test - mu

        # Economy SVD on centered train features.
        # For d=512 this is inexpensive even on the full training split.
        try:
            _, S, Vt = np.linalg.svd(X_train_c, full_matrices=False)
        except np.linalg.LinAlgError as exc:
            logger.warning(f"  [{run_name}] epoch {epoch}: SVD failed ({exc})  -- skipping.")
            continue

        # np.linalg.svd returns singular values sorted descending.
        k = select_k_threshold(S, args.threshold_mode, args.threshold_value)

        # Project to principal-component coordinates.
        # scores[:, i] is the coordinate along the i-th right singular vector.
        # Training probes on scores is equivalent to training on the
        # reconstructed representations for linear readouts, and avoids
        # materialising a large (n x d) reconstruction matrix.
        scores_train = X_train_c @ Vt.T   # (n_train, d)
        scores_test = X_test_c @ Vt.T     # (n_test,  d)

        main_train = scores_train[:, :k]
        main_test = scores_test[:, :k]
        residual_train = scores_train[:, k:]
        residual_test = scores_test[:, k:]

        # --- Spectral summary metrics ---
        metrics = compute_spectral_metrics(S, k, args.threshold_mode, args.threshold_value)
        summary_rows.append({"run_name": run_name, "lr": lr, "seed": seed, "epoch": epoch, **metrics})

        # --- Long-format singular values ---
        sv_rows.extend(build_sv_rows(run_name, lr, seed, epoch, S))

        # --- Probe readout on each component ---
        components = {
            "full": (scores_train, scores_test),
            "main": (main_train, main_test),
            "residual": (residual_train, residual_test),
        }
        for comp_name, (Xtr, Xte) in components.items():
            if Xtr.shape[1] == 0:
                # residual is empty when k == d; skip silently.
                continue
            try:
                train_acc, test_acc, fit_secs = fit_probe(
                    Xtr, y_train, Xte, y_test, seed=args.seed
                )
                logger.info(
                    f"    {comp_name:8s}  train={train_acc:.4f}  test={test_acc:.4f}"
                    f"  dim={Xtr.shape[1]}  k_main={k}  fit={fit_secs:.1f}s"
                )
            except Exception:
                err = traceback.format_exc().strip().split("\n")[-1][:300]
                logger.error(f"    {comp_name:8s}  FAILED: {err}")
                train_acc = test_acc = float("nan")

            probe_rows.append({
                "run_name": run_name,
                "lr": lr,
                "seed": seed,
                "epoch": epoch,
                "component": comp_name,
                "probe": args.probe,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "k_main": k,
                "threshold_mode": args.threshold_mode,
                "threshold_value": args.threshold_value,
            })

    # --- Save per-run CSVs ---
    _write_csv(summary_rows, summary_path, SUMMARY_FIELDS, logger,
               label=f"[{run_name}] summary")
    _write_csv(probe_rows, probe_path, PROBE_FIELDS, logger,
               label=f"[{run_name}] probe_results")
    _write_csv(sv_rows, sv_path, SV_FIELDS, logger,
               label=f"[{run_name}] singular_values")

    return summary_rows, probe_rows, sv_rows


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _write_csv(
    rows: list,
    path: str,
    fields: list,
    logger: logging.Logger,
    label: str = "",
) -> None:
    if not rows:
        logger.warning(f"{label}: no rows to write, skipping {path}")
        return
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"{label}: saved {len(rows)} rows  ->  {path}")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_and_save(
    all_summary: list,
    all_probe: list,
    all_sv: list,
    agg_dir: str,
    logger: logging.Logger,
):
    """Concatenate per-run rows and write aggregate CSVs."""
    os.makedirs(agg_dir, exist_ok=True)
    summary_path = os.path.join(agg_dir, "spectral_component_summary.csv")
    probe_path = os.path.join(agg_dir, "spectral_probe_results.csv")
    sv_path = os.path.join(agg_dir, "spectral_singular_values.csv")
    _write_csv(all_summary, summary_path, SUMMARY_FIELDS, logger, label="[aggregate] summary")
    _write_csv(all_probe, probe_path, PROBE_FIELDS, logger, label="[aggregate] probe_results")
    _write_csv(all_sv, sv_path, SV_FIELDS, logger, label="[aggregate] singular_values")
    return summary_path, probe_path, sv_path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_LR_LABELS = {0.05: "LR = 0.05", 0.10: "LR = 0.10", 0.20: "LR = 0.20"}

_COMPONENT_STYLES = {
    "full":     dict(color="#1f77b4", linestyle="-",  linewidth=1.5, label="full"),
    "main":     dict(color="#2ca02c", linestyle="--", linewidth=1.5, label="main (dominant)"),
    "residual": dict(color="#d62728", linestyle=":",  linewidth=1.5, label="residual"),
}


def _make_lr_fig(unique_lrs: list, figsize_per=(5, 4)):
    n = len(unique_lrs)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per[0] * n, figsize_per[1]), squeeze=False)
    return fig, axes[0]


def plot_probe_trajectories(
    probe_df: pd.DataFrame, plot_dir: str, logger: logging.Logger
) -> None:
    """
    Central plot: logistic probe test accuracy on full/main/residual components
    over epochs, mean +/- 1 std across seeds, one panel per learning rate.
    """
    if probe_df.empty:
        logger.warning("Probe data empty; skipping probe trajectory plot.")
        return

    unique_lrs = sorted(probe_df["lr"].unique())
    fig, axes = _make_lr_fig(unique_lrs)

    for ax, lr in zip(axes, unique_lrs):
        sub = probe_df[probe_df["lr"] == lr]
        for comp, style in _COMPONENT_STYLES.items():
            grp = sub[sub["component"] == comp].groupby("epoch")["test_acc"]
            if grp.ngroups == 0:
                continue
            mean = grp.mean()
            std = grp.std(ddof=1).fillna(0.0)
            epochs = mean.index.values
            ax.plot(epochs, mean.values, **{k: v for k, v in style.items() if k != "label"},
                    label=style["label"])
            ax.fill_between(epochs, mean.values - std.values, mean.values + std.values,
                            alpha=0.15, color=style["color"])
        ax.set_title(_LR_LABELS.get(lr, f"LR={lr}"))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test accuracy")
        ax.legend(fontsize=8)
        ax.grid(True, linewidth=0.4)

    fig.suptitle(
        "Spectral component probe accuracy (logistic regression)", fontsize=11
    )
    fig.tight_layout()
    out = os.path.join(plot_dir, "probe_trajectories.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")


def plot_k_main_over_time(
    summary_df: pd.DataFrame, plot_dir: str, logger: logging.Logger
) -> None:
    """
    Number of dominant singular directions k over epochs,
    mean +/- 1 std across seeds, one panel per learning rate.
    """
    if summary_df.empty:
        logger.warning("Summary data empty; skipping k_main trajectory plot.")
        return

    unique_lrs = sorted(summary_df["lr"].unique())
    fig, axes = _make_lr_fig(unique_lrs)

    for ax, lr in zip(axes, unique_lrs):
        grp = summary_df[summary_df["lr"] == lr].groupby("epoch")["k_main"]
        mean = grp.mean()
        std = grp.std(ddof=1).fillna(0.0)
        ax.plot(mean.index.values, mean.values, color="#7f7f7f", linewidth=1.5)
        ax.fill_between(
            mean.index.values,
            mean.values - std.values,
            mean.values + std.values,
            alpha=0.2, color="#7f7f7f",
        )
        ax.set_title(_LR_LABELS.get(lr, f"LR={lr}"))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("k (dominant singular directions)")
        ax.grid(True, linewidth=0.4)

    fig.suptitle(
        "Number of dominant singular directions (k) over training", fontsize=11
    )
    fig.tight_layout()
    out = os.path.join(plot_dir, "k_main_over_time.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")


def plot_spectral_mass_over_time(
    summary_df: pd.DataFrame, plot_dir: str, logger: logging.Logger
) -> None:
    """
    Main singular mass and main energy mass over epochs,
    mean +/- 1 std across seeds, one panel per learning rate.
    """
    if summary_df.empty:
        logger.warning("Summary data empty; skipping spectral mass plot.")
        return

    unique_lrs = sorted(summary_df["lr"].unique())
    fig, axes = _make_lr_fig(unique_lrs)

    mass_series = [
        ("main_singular_mass", "#2ca02c", "singular mass"),
        ("main_energy_mass",   "#9467bd", "energy mass"),
    ]

    for ax, lr in zip(axes, unique_lrs):
        sub = summary_df[summary_df["lr"] == lr]
        for col, color, label in mass_series:
            grp = sub.groupby("epoch")[col]
            mean = grp.mean()
            std = grp.std(ddof=1).fillna(0.0)
            ax.plot(mean.index.values, mean.values, color=color, linewidth=1.5, label=label)
            ax.fill_between(
                mean.index.values,
                mean.values - std.values,
                mean.values + std.values,
                alpha=0.15, color=color,
            )
        ax.set_title(_LR_LABELS.get(lr, f"LR={lr}"))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Fraction of total mass in main component")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(True, linewidth=0.4)

    fig.suptitle("Main spectral mass and energy mass over training", fontsize=11)
    fig.tight_layout()
    out = os.path.join(plot_dir, "spectral_mass_over_time.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")


def plot_singular_spectrum_snapshots(
    sv_df: pd.DataFrame, plot_dir: str, logger: logging.Logger
) -> None:
    """
    Optional: singular value spectra at selected epochs on a log scale.
    Curves are averaged across seeds per LR to reduce clutter.
    """
    if sv_df.empty:
        logger.warning("Singular values data empty; skipping spectrum snapshot plot.")
        return

    snap_epochs = sorted(set(sv_df["epoch"].unique()) & SNAPSHOT_EPOCHS)
    if not snap_epochs:
        logger.warning("No snapshot epochs present in singular values data; skipping.")
        return

    unique_lrs = sorted(sv_df["lr"].unique())
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(snap_epochs), vmax=max(snap_epochs))
    fig, axes = _make_lr_fig(unique_lrs)

    for ax, lr in zip(axes, unique_lrs):
        sub = sv_df[sv_df["lr"] == lr]
        for ep in snap_epochs:
            ep_data = sub[sub["epoch"] == ep]
            if ep_data.empty:
                continue
            mean_sv = ep_data.groupby("component_index")["singular_value"].mean().values
            ax.semilogy(
                np.arange(len(mean_sv)), mean_sv,
                color=cmap(norm(ep)), linewidth=1.0, label=f"ep {ep}",
            )
        ax.set_title(_LR_LABELS.get(lr, f"LR={lr}"))
        ax.set_xlabel("Component index")
        ax.set_ylabel("Singular value (log scale)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, linewidth=0.4, which="both")

    fig.suptitle("Singular value spectrum at selected epochs", fontsize=11)
    fig.tight_layout()
    out = os.path.join(plot_dir, "singular_spectrum_snapshots.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")


def create_plots(
    summary_path: str,
    probe_path: str,
    sv_path: str,
    plot_dir: str,
    logger: logging.Logger,
) -> None:
    """Load aggregate CSVs and produce all spectral analysis plots."""
    os.makedirs(plot_dir, exist_ok=True)
    try:
        probe_df = pd.read_csv(probe_path)
        summary_df = pd.read_csv(summary_path)
        sv_df = pd.read_csv(sv_path)
    except Exception as exc:
        logger.error(f"Could not load aggregate CSVs for plotting: {exc}")
        return

    plot_probe_trajectories(probe_df, plot_dir, logger)
    plot_k_main_over_time(summary_df, plot_dir, logger)
    plot_spectral_mass_over_time(summary_df, plot_dir, logger)
    plot_singular_spectrum_snapshots(sv_df, plot_dir, logger)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    agg_dir = os.path.join(args.repo_results_dir, "_aggregate")
    plot_dir = os.path.join(agg_dir, "plots", "spectral")
    log_path = os.path.join(agg_dir, "logs", "spectral_components.log")

    logger = setup_logging(log_path)
    logger.info("=" * 60)
    logger.info("spectral_components.py  start")
    logger.info(f"  base_dir          = {args.base_dir}")
    logger.info(f"  repo_results_dir  = {args.repo_results_dir}")
    logger.info(f"  runs pattern      = {args.runs}")
    logger.info(f"  threshold_mode    = {args.threshold_mode}")
    logger.info(f"  threshold_value   = {args.threshold_value}")
    logger.info(f"  probe             = {args.probe}")
    logger.info(f"  overwrite         = {args.overwrite}")
    logger.info(f"  seed              = {args.seed}")
    logger.info("=" * 60)

    run_dirs = discover_runs(args.base_dir, args.runs)
    if not run_dirs:
        logger.error(
            f"No run directories found matching '{args.runs}' under {args.base_dir}."
        )
        sys.exit(1)

    logger.info(f"Discovered {len(run_dirs)} run(s).")

    all_summary: list = []
    all_probe: list = []
    all_sv: list = []
    n_ok = 0
    n_skip = 0

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        parsed = parse_run_name(run_name)
        if parsed is None:
            logger.warning(
                f"Cannot parse lr/seed from run name '{run_name}'; skipping."
            )
            n_skip += 1
            continue
        lr, seed = parsed

        try:
            s_rows, p_rows, sv_rows = process_run(
                run_dir, run_name, lr, seed, args, logger
            )
        except Exception:
            logger.error(
                f"[{run_name}] Unexpected error:\n{traceback.format_exc()}"
            )
            n_skip += 1
            continue

        if s_rows is None:
            n_skip += 1
            continue

        all_summary.extend(s_rows)
        all_probe.extend(p_rows)
        all_sv.extend(sv_rows)
        n_ok += 1

    logger.info(f"Runs completed: {n_ok}  skipped: {n_skip}")

    if not all_summary:
        logger.error("No results collected; cannot write aggregate files.")
        sys.exit(1)

    summary_path, probe_path, sv_path = aggregate_and_save(
        all_summary, all_probe, all_sv, agg_dir, logger
    )

    logger.info("Creating plots...")
    create_plots(summary_path, probe_path, sv_path, plot_dir, logger)

    logger.info("spectral_components.py  done.")


if __name__ == "__main__":
    main()
