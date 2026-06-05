"""
analysis/build_master_table.py — Merge all per-epoch metrics into master_trajectory.csv.

The master trajectory is the primary analysis artifact of the fullstudy experiment.
Every major metric is aligned by epoch and merged into a single table so that
stabilization (CKA), sufficiency (probe accuracy), and geometry (NC) can be
compared row by row without manual epoch matching.

Source files (all in config["paths"]["results"]):
  train_metrics.csv        — LR, loss, accuracy every training epoch (required)
  cka_summary.csv          — CKA metrics at checkpoint epochs (required)
  probe_results_wide.csv   — surrogate probe test accuracies (required)
  neural_collapse.csv      — NC1–NC4 at checkpoint epochs (required)
  entk_results.csv         — eNTK dynamics (optional; merged if present)

Output:
  master_trajectory.csv    — one row per stored checkpoint epoch

Required output columns:
  epoch
  lr, train_loss, train_acc, network_test_acc
  local_cka_change, cka_to_final, mean_future_cka, below_tau
  nc1, log10_nc1, nc2_etf_deviation, nc3_weight_mean_alignment, nc4_ncm_disagreement
  logistic_acc, linear_svm_acc, rbf_svm_acc, rf_acc, lightgbm_acc

Derived sufficiency columns (added automatically):
  probe_gap_to_final(t) = final_probe_accuracy - probe_accuracy(t)
    logistic_gap_to_final, linear_svm_gap_to_final, rbf_svm_gap_to_final,
    rf_gap_to_final, lightgbm_gap_to_final

  probe_relative_to_final_network(t) = probe_accuracy(t) / final_network_test_accuracy
    logistic_relative_to_final_network, linear_svm_relative_to_final_network,
    rbf_svm_relative_to_final_network, rf_relative_to_final_network,
    lightgbm_relative_to_final_network

Usage:
  python analysis/build_master_table.py \\
      --config configs/resnet18_cifar10_200_fullstudy.yaml
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Logging and config
# ---------------------------------------------------------------------------

def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("build_master_table")
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
# Source CSV loading (graceful on missing files)
# ---------------------------------------------------------------------------

def load_source_csv(
    path: str,
    label: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Load a source CSV as a DataFrame.

    Returns an empty DataFrame with only an 'epoch' column if the file is
    missing — the caller then left-joins this against the master epoch list,
    producing NaN values for that metric group rather than crashing.
    """
    if not os.path.exists(path):
        logger.warning(
            f"{label}: file not found: {path}. "
            "Those metric columns will be NaN in the master table."
        )
        return pd.DataFrame({"epoch": pd.Series([], dtype=int)})

    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.error(f"{label}: failed to read {path}: {e}")
        return pd.DataFrame({"epoch": pd.Series([], dtype=int)})

    if "epoch" not in df.columns:
        logger.error(f"{label}: 'epoch' column missing in {path}. Treating as missing.")
        return pd.DataFrame({"epoch": pd.Series([], dtype=int)})

    df["epoch"] = df["epoch"].astype(int)
    logger.info(f"{label}: loaded {len(df)} rows  path={path}")
    return df


def require_at_least_one_required_source(
    results_dir: str,
    logger: logging.Logger,
) -> None:
    """Fail if no required fullstudy source CSV exists yet."""
    required_source_files = [
        "train_metrics.csv",
        "cka_summary.csv",
        "probe_results_wide.csv",
        "neural_collapse.csv",
    ]
    existing = [
        filename for filename in required_source_files
        if os.path.exists(os.path.join(results_dir, filename))
    ]

    if existing:
        missing = sorted(set(required_source_files) - set(existing))
        if missing:
            logger.warning(
                f"Required source CSVs missing; continuing with NaNs for those groups: {missing}"
            )
        return

    logger.error(
        "No required source CSVs were found. Refusing to create an empty "
        "master_trajectory.csv.\n"
        f"  Results dir: {results_dir}\n"
        f"  Required sources: {required_source_files}\n"
        "Run at least one upstream fullstudy step before building the master table."
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Column selection helpers
# ---------------------------------------------------------------------------

def select_columns(df: pd.DataFrame, desired_cols: list) -> pd.DataFrame:
    """
    Return a copy of df containing only the columns in desired_cols that exist.
    'epoch' is always kept. Columns absent from df are silently dropped.
    """
    available = [c for c in desired_cols if c in df.columns]
    always_include = ["epoch"]
    cols_to_keep = always_include + [c for c in available if c not in always_include]
    return df[cols_to_keep].copy()


# ---------------------------------------------------------------------------
# Derived sufficiency columns
# ---------------------------------------------------------------------------

def add_derived_sufficiency_columns(
    master_df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Add gap-to-final and relative-to-network-final columns for each probe.

    probe_gap_to_final(t)
        = probe_accuracy_at_final_epoch − probe_accuracy(t)
        A positive gap means the probe at epoch t underperforms the final probe.
        Gap → 0 as representations approach their final state.

    probe_relative_to_final_network(t)
        = probe_accuracy(t) / network_test_accuracy_at_final_epoch
        Values > 1 mean the probe already exceeds the final network accuracy.

    "final epoch" is defined as the row with the largest epoch value.
    """
    final_epoch = int(master_df["epoch"].max())
    final_mask = master_df["epoch"] == final_epoch
    final_row = master_df.loc[final_mask].iloc[0]

    logger.info(f"Derived columns: final epoch = {final_epoch}")

    # Final network test accuracy — used as the denominator for relative columns
    final_network_test_acc = pd.to_numeric(
        final_row.get("network_test_acc", np.nan), errors="coerce"
    )
    logger.info(f"  Final network test accuracy = {final_network_test_acc}")

    probe_column_specs = [
        ("logistic_acc",    "logistic_gap_to_final",    "logistic_relative_to_final_network"),
        ("linear_svm_acc",  "linear_svm_gap_to_final",  "linear_svm_relative_to_final_network"),
        ("rbf_svm_acc",     "rbf_svm_gap_to_final",     "rbf_svm_relative_to_final_network"),
        ("rf_acc",          "rf_gap_to_final",          "rf_relative_to_final_network"),
        ("lightgbm_acc",    "lightgbm_gap_to_final",    "lightgbm_relative_to_final_network"),
    ]

    for acc_col, gap_col, rel_col in probe_column_specs:
        if acc_col not in master_df.columns:
            # Probe was not fitted (entire metric group missing)
            master_df[gap_col] = np.nan
            master_df[rel_col] = np.nan
            logger.info(f"  {acc_col} absent — {gap_col} and {rel_col} set to NaN")
            continue

        probe_accuracies = pd.to_numeric(master_df[acc_col], errors="coerce")

        final_probe_acc = pd.to_numeric(
            final_row.get(acc_col, np.nan), errors="coerce"
        )

        # gap_to_final = final_probe_acc - probe_acc(t)
        gap_values = final_probe_acc - probe_accuracies
        master_df[gap_col] = gap_values

        # relative_to_final_network = probe_acc(t) / final_network_test_acc
        if np.isnan(final_network_test_acc):
            relative_values = np.full(len(master_df), np.nan)
        else:
            relative_values = probe_accuracies / final_network_test_acc

        master_df[rel_col] = relative_values

        logger.info(
            f"  {acc_col}: final_probe={final_probe_acc:.4f}"
            f"  gap_range=[{gap_values.min():.4f}, {gap_values.max():.4f}]"
        )

    return master_df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_master_table(
    master_df: pd.DataFrame,
    checkpoint_epochs: list,
    required_columns: list,
    logger: logging.Logger,
) -> bool:
    """
    Run basic sanity checks on the merged master table. Returns True if all
    checks pass; logs errors and returns False on any failure.

    Checks:
    1. 'epoch' column exists
    2. No duplicate epochs
    3. One row per checkpoint epoch
    4. All required columns present
    5. At least one derived sufficiency column present
    """
    passed = True

    # Check 1: epoch column exists
    if "epoch" not in master_df.columns:
        logger.error("Validation FAIL: 'epoch' column is missing from master table")
        return False
    logger.info("Validation 1/5: 'epoch' column present")

    # Check 2: no duplicate epochs
    n_unique_epochs = master_df["epoch"].nunique()
    n_total_rows = len(master_df)
    if n_unique_epochs != n_total_rows:
        logger.error(
            f"Validation FAIL: duplicate epochs found "
            f"({n_total_rows} rows but only {n_unique_epochs} unique epochs)"
        )
        passed = False
    else:
        logger.info(f"Validation 2/5: no duplicate epochs ({n_total_rows} rows)")

    # Check 3: one row per checkpoint epoch
    master_epochs = sorted(master_df["epoch"].tolist())
    if master_epochs != sorted(checkpoint_epochs):
        missing = sorted(set(checkpoint_epochs) - set(master_epochs))
        extra = sorted(set(master_epochs) - set(checkpoint_epochs))
        logger.error(
            f"Validation FAIL: epoch mismatch. "
            f"Missing: {missing}  Extra: {extra}"
        )
        passed = False
    else:
        logger.info(f"Validation 3/5: all {len(checkpoint_epochs)} checkpoint epochs present")

    # Check 4: required columns present
    missing_cols = [c for c in required_columns if c not in master_df.columns]
    if missing_cols:
        logger.warning(
            f"Validation WARNING: {len(missing_cols)} required columns absent "
            f"(data not yet available): {missing_cols}"
        )
        # This is a warning, not a failure — the column will be NaN
    else:
        logger.info(f"Validation 4/5: all {len(required_columns)} required columns present")

    # Check 5: at least one derived sufficiency column exists
    derived_gap_cols = [
        "logistic_gap_to_final", "linear_svm_gap_to_final",
        "rbf_svm_gap_to_final", "rf_gap_to_final", "lightgbm_gap_to_final",
    ]
    has_derived = any(c in master_df.columns for c in derived_gap_cols)
    if not has_derived:
        logger.error("Validation FAIL: no derived gap-to-final columns found")
        passed = False
    else:
        n_derived = sum(1 for c in derived_gap_cols if c in master_df.columns)
        logger.info(f"Validation 5/5: {n_derived}/{len(derived_gap_cols)} gap-to-final columns present")

    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Parse arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Merge all per-epoch metrics into results/master_trajectory.csv. "
            "Run after: train.py, cka_trajectory.py, run_probes.py, neural_collapse.py."
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
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 2. Load config and resolve paths
    # ------------------------------------------------------------------
    config = load_config(args.config)

    results_dir = args.results_dir or config["paths"]["results"]
    logs_dir = config["paths"].get("logs", results_dir)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Logging
    # ------------------------------------------------------------------
    log_path = os.path.join(logs_dir, "build_master_table.log")
    logger = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("Building Master Trajectory Table")
    logger.info("=" * 70)
    logger.info(f"Config:      {args.config}")
    logger.info(f"Results dir: {results_dir}")

    # ------------------------------------------------------------------
    # 4. Determine the definitive checkpoint epoch list from config
    # ------------------------------------------------------------------
    checkpoint_epochs = sorted(config["checkpoint_epochs"])
    logger.info(f"Checkpoint epochs from config: {len(checkpoint_epochs)} epochs")
    logger.info(f"  Range: {checkpoint_epochs[0]} to {checkpoint_epochs[-1]}")

    # ------------------------------------------------------------------
    # 5. Load all source CSVs
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("Loading source CSVs...")
    require_at_least_one_required_source(results_dir=results_dir, logger=logger)

    train_df = load_source_csv(
        path=os.path.join(results_dir, "train_metrics.csv"),
        label="train_metrics",
        logger=logger,
    )
    cka_df = load_source_csv(
        path=os.path.join(results_dir, "cka_summary.csv"),
        label="cka_summary",
        logger=logger,
    )
    probe_df = load_source_csv(
        path=os.path.join(results_dir, "probe_results_wide.csv"),
        label="probe_results_wide",
        logger=logger,
    )
    nc_df = load_source_csv(
        path=os.path.join(results_dir, "neural_collapse.csv"),
        label="neural_collapse",
        logger=logger,
    )

    # Optional: eNTK results (not required; merged gracefully if present)
    entk_path = os.path.join(results_dir, "entk_results.csv")
    entk_df = load_source_csv(
        path=entk_path,
        label="entk_results (optional)",
        logger=logger,
    )

    # ------------------------------------------------------------------
    # 6. Build master DataFrame starting from checkpoint epoch list
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("Merging sources by epoch...")

    master_df = pd.DataFrame({"epoch": checkpoint_epochs})
    master_df["epoch"] = master_df["epoch"].astype(int)

    # ── train_metrics: rename test_acc → network_test_acc, drop test_loss ─
    train_cols_needed = ["epoch", "lr", "train_loss", "train_acc", "test_acc"]
    train_subset = select_columns(train_df, train_cols_needed)
    if "test_acc" in train_subset.columns:
        train_subset = train_subset.rename(columns={"test_acc": "network_test_acc"})
    master_df = master_df.merge(train_subset, on="epoch", how="left")
    logger.info(f"  After train_metrics merge: {master_df.shape}")

    # ── cka_summary: keep only analysis-relevant columns ──────────────────
    cka_cols_needed = [
        "epoch", "local_cka_change", "cka_to_final", "mean_future_cka", "below_tau",
    ]
    cka_subset = select_columns(cka_df, cka_cols_needed)
    master_df = master_df.merge(cka_subset, on="epoch", how="left")
    logger.info(f"  After cka_summary merge: {master_df.shape}")

    # ── neural_collapse: all metric columns ───────────────────────────────
    nc_cols_needed = [
        "epoch", "nc1", "log10_nc1", "nc2_etf_deviation",
        "nc3_weight_mean_alignment", "nc4_ncm_disagreement",
    ]
    nc_subset = select_columns(nc_df, nc_cols_needed)
    master_df = master_df.merge(nc_subset, on="epoch", how="left")
    logger.info(f"  After neural_collapse merge: {master_df.shape}")

    # ── probe_results_wide: all probe accuracy columns ────────────────────
    probe_cols_needed = [
        "epoch", "logistic_acc", "linear_svm_acc", "rbf_svm_acc",
        "rf_acc", "lightgbm_acc",
    ]
    probe_subset = select_columns(probe_df, probe_cols_needed)
    master_df = master_df.merge(probe_subset, on="epoch", how="left")
    logger.info(f"  After probe_results_wide merge: {master_df.shape}")

    # ── eNTK: optional — merge all columns if file exists ─────────────────
    if len(entk_df) > 0:
        entk_subset = entk_df.copy()
        master_df = master_df.merge(entk_subset, on="epoch", how="left")
        logger.info(f"  After entk_results merge: {master_df.shape}")
    else:
        logger.info("  eNTK: skipped (file absent or empty)")

    # ------------------------------------------------------------------
    # 7. Enforce the required column order
    # ------------------------------------------------------------------
    required_columns_ordered = [
        "epoch",
        # Training dynamics
        "lr", "train_loss", "train_acc", "network_test_acc",
        # CKA trajectory
        "local_cka_change", "cka_to_final", "mean_future_cka", "below_tau",
        # Neural Collapse
        "nc1", "log10_nc1", "nc2_etf_deviation",
        "nc3_weight_mean_alignment", "nc4_ncm_disagreement",
        # Surrogate probes (test accuracy)
        "logistic_acc", "linear_svm_acc", "rbf_svm_acc", "rf_acc", "lightgbm_acc",
    ]

    # Columns that exist but are not in the required list (e.g. eNTK, extras)
    extra_columns = [c for c in master_df.columns if c not in required_columns_ordered]

    # Build final column order: required first (if present), then extras
    final_columns = (
        [c for c in required_columns_ordered if c in master_df.columns]
        + extra_columns
    )
    master_df = master_df[final_columns]

    # ------------------------------------------------------------------
    # 8. Add derived sufficiency columns
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("Adding derived sufficiency columns...")
    master_df = add_derived_sufficiency_columns(master_df, logger)

    # ------------------------------------------------------------------
    # 9. Validate
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("Validating master table...")
    valid = validate_master_table(
        master_df=master_df,
        checkpoint_epochs=checkpoint_epochs,
        required_columns=required_columns_ordered,
        logger=logger,
    )
    if not valid:
        logger.error("Validation failed — master table may be incomplete. Check log.")

    # ------------------------------------------------------------------
    # 10. Save
    # ------------------------------------------------------------------
    output_path = os.path.join(results_dir, "master_trajectory.csv")
    master_df.to_csv(output_path, index=False)
    logger.info(f"Saved master_trajectory.csv: {master_df.shape}  ->  {output_path}")

    # ------------------------------------------------------------------
    # 11. Summary
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("Master trajectory build complete.")
    logger.info(f"  Rows: {len(master_df)} (one per checkpoint epoch)")
    logger.info(f"  Columns: {len(master_df.columns)}")
    logger.info(f"  Epoch range: {master_df['epoch'].min()} to {master_df['epoch'].max()}")

    # Log NaN counts per column group for quick data completeness check
    nan_counts = master_df.isna().sum()
    cols_with_all_nan = [c for c in master_df.columns if nan_counts[c] == len(master_df)]
    if cols_with_all_nan:
        logger.warning(f"  Columns fully empty (source file missing): {cols_with_all_nan}")

    logger.info(f"  Output: {output_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
