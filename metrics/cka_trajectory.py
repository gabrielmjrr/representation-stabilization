"""
metrics/cka_trajectory.py — Full CKA trajectory analysis for the fullstudy pipeline.

Reads pre-extracted feature files from the cka_subset directory produced by
features/extract_features.py, then computes temporal CKA across all stored
checkpoint epochs and writes four output CSVs.

Difference from metrics/cka.py
  metrics/cka.py         — reads flat activations_epoch_XXXX.npy produced by
                            train.py's inline extraction; used for the previous
                            experiment pipeline.
  metrics/cka_trajectory.py — reads structured features_epoch_XXXX.npy produced
                            by features/extract_features.py; used for the
                            fullstudy pipeline.  Adds cka_to_final,
                            mean_future_cka, and the matrix output.

The core CKA math (center_gram_matrix, compute_hsic, compute_linear_cka) is
identical to metrics/cka.py and is duplicated here to keep this script
self-contained and readable top-to-bottom without import tracing.

Modes
  consecutive  Compute CKA only between adjacent checkpoint pairs.
               Produces: cka_consecutive.csv

  pairwise     Compute CKA between ALL epoch pairs (full N×N matrix).
               Produces: cka_consecutive.csv
                         cka_pairwise_long.csv
                         cka_matrix.csv

  summary      Derive per-epoch summary metrics from the full pairwise matrix.
  (default)    Produces: cka_consecutive.csv
                         cka_pairwise_long.csv
                         cka_matrix.csv
                         cka_summary.csv

Summary metrics (cka_summary.csv, one row per checkpoint epoch)
  local_cka_change  = 1 - CKA(t, previous_checkpoint)
                      How much representations changed since the last checkpoint.
                      NaN for the first epoch (no previous checkpoint exists).
  cka_to_final      = CKA(t, final_epoch)
                      Similarity of epoch-t representation to the final converged
                      state.  Rises toward 1 as training proceeds.
  mean_future_cka   = mean over all future checkpoints t' > t of CKA(t, t')
                      How representative epoch-t's representation is of the
                      network's future states.  NaN for the final epoch.
  below_tau         = int(local_cka_change < tau)
                      1 if this pair is "stable" per the tau threshold.
                      tau is recorded for reference; it is not a trigger here.

Usage
  python metrics/cka_trajectory.py --config configs/resnet18_cifar10_200_fullstudy.yaml
  python metrics/cka_trajectory.py --config ... --mode consecutive
  python metrics/cka_trajectory.py --config ... --mode pairwise
  python metrics/cka_trajectory.py --config ... --mode summary   [default]
  python metrics/cka_trajectory.py --config ... --features-dir /path/to/cka_subset
"""

import argparse
import csv
import logging
import os
import sys

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("cka_trajectory")
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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ---------------------------------------------------------------------------
# Feature file discovery and loading
# ---------------------------------------------------------------------------

def find_feature_epochs(features_dir: str) -> list:
    """
    Discover all features_epoch_XXXX.npy files in features_dir.
    Returns a list of epoch numbers sorted ascending.
    Filename convention matches features/extract_features.py.
    """
    epochs = []
    for filename in os.listdir(features_dir):
        if not filename.startswith("features_epoch_"):
            continue
        if not filename.endswith(".npy"):
            continue
        epoch_str = filename.replace("features_epoch_", "").replace(".npy", "")
        epochs.append(int(epoch_str))
    epochs.sort()
    return epochs


def load_all_features(
    features_dir: str,
    epochs: list,
    logger: logging.Logger,
) -> list:
    """
    Load all feature files into memory at once.

    Loading all files before the pair loop avoids O(n²) redundant disk reads.
    For 40 checkpoints × 2040 samples × 512 features × float32 this is ~167 MB,
    well within cluster RAM.

    Returns:
        List of (epoch, features_array) pairs sorted by epoch.
        features_array shape: (n_samples, n_features)
    """
    loaded = []
    for epoch in epochs:
        filename = f"features_epoch_{epoch:04d}.npy"
        path = os.path.join(features_dir, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Feature file not found: {path}\n"
                "Run features/extract_features.py first."
            )

        features = np.load(path)
        loaded.append((epoch, features))
        logger.info(f"  Loaded epoch {epoch:4d}  shape={features.shape}  dtype={features.dtype}")

    return loaded


# ---------------------------------------------------------------------------
# CKA computation — Kornblith et al. 2019, linear kernel
# Same math as metrics/cka.py; duplicated here for self-containedness.
# ---------------------------------------------------------------------------

def center_gram_matrix(gram_matrix: np.ndarray) -> np.ndarray:
    """
    Double-center a Gram matrix: K_c = H K H  where  H = I - (1/n) 11^T.

    Equivalent to subtracting each row mean, each column mean, and adding
    back the grand mean.  Required before computing HSIC (Kornblith et al.
    2019, Appendix A).
    """
    row_means = gram_matrix.mean(axis=1, keepdims=True)   # (n, 1)
    col_means = gram_matrix.mean(axis=0, keepdims=True)   # (1, n)
    grand_mean = gram_matrix.mean()                        # scalar

    gram_matrix_centered = gram_matrix - row_means - col_means + grand_mean
    return gram_matrix_centered


def compute_hsic(K_centered: np.ndarray, L_centered: np.ndarray) -> float:
    """
    Biased HSIC from pre-centered Gram matrices.
    HSIC(K, L) = (1 / (n-1)^2) * sum_ij K_c[i,j] * L_c[i,j]

    Uses the Frobenius inner product instead of the full matrix product to
    avoid an O(n^3) matmul.  The bias is O(1/n) — negligible at n=2040.
    """
    n_samples = K_centered.shape[0]
    normalization = float((n_samples - 1) ** 2)
    frobenius_inner_product = float(np.sum(K_centered * L_centered))
    hsic_value = frobenius_inner_product / normalization
    return hsic_value


def compute_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between representation matrices X ∈ R^{n×p} and Y ∈ R^{n×q}.

    CKA = HSIC(K_c, L_c) / sqrt(HSIC(K_c, K_c) * HSIC(L_c, L_c))
    where K = X X^T  and  L = Y Y^T  (linear Gram matrices).

    Returns a float in [0, 1].  Returns 0.0 when either representation has
    near-zero variance (denominator < 1e-10).

    This is within-network temporal CKA (comparing the same network at
    two different training epochs), not cross-network CKA.
    """
    gram_X = X @ X.T   # (n, n)
    gram_Y = Y @ Y.T   # (n, n)

    gram_X_centered = center_gram_matrix(gram_X)
    gram_Y_centered = center_gram_matrix(gram_Y)

    hsic_XY = compute_hsic(gram_X_centered, gram_Y_centered)
    hsic_XX = compute_hsic(gram_X_centered, gram_X_centered)
    hsic_YY = compute_hsic(gram_Y_centered, gram_Y_centered)

    denominator = np.sqrt(hsic_XX * hsic_YY)

    if denominator < 1e-10:
        # Near-zero variance in one or both representations.
        # Can occur at epoch 0 before meaningful gradient updates.
        return 0.0

    cka_value = hsic_XY / denominator
    return float(cka_value)


# ---------------------------------------------------------------------------
# Pairwise CKA matrix
# ---------------------------------------------------------------------------

def compute_pairwise_cka_matrix(
    all_features: list,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Compute the full N×N symmetric CKA matrix.

    matrix[i][j] = CKA(all_features[i][1], all_features[j][1])

    Only the upper triangle is computed; the lower triangle is filled by
    symmetry.  The diagonal is always 1.0.

    Returns:
        matrix: (N, N) float64 ndarray, symmetric, diagonal = 1.
    """
    n = len(all_features)
    n_pairs_upper_triangle = n * (n + 1) // 2

    matrix = np.zeros((n, n), dtype=np.float64)

    pair_count = 0
    for i in range(n):
        epoch_i, features_i = all_features[i]

        for j in range(i, n):
            epoch_j, features_j = all_features[j]

            if i == j:
                # A representation is always identical to itself
                cka_val = 1.0
            else:
                cka_val = compute_linear_cka(features_i, features_j)

            matrix[i][j] = cka_val
            matrix[j][i] = cka_val   # fill lower triangle by symmetry

            pair_count += 1
            logger.info(
                f"  Pair {pair_count:4d}/{n_pairs_upper_triangle}"
                f"  (epoch {epoch_i:4d}, {epoch_j:4d})"
                f"  CKA = {cka_val:.4f}"
            )

    return matrix


# ---------------------------------------------------------------------------
# Derive output rows from the pairwise matrix
# ---------------------------------------------------------------------------

def derive_consecutive_rows(
    epochs: list,
    n_samples: int,
    matrix: np.ndarray,
    tau: float,
) -> list:
    """
    Build rows for cka_consecutive.csv from adjacent checkpoint pairs.

    Each row covers one (epoch_prev → epoch_curr) step in the sorted epoch list.
    local_cka_change = 1 - CKA(curr, prev).
    below_tau = 1 if local_cka_change < tau.
    """
    rows = []
    for i in range(1, len(epochs)):
        epoch_prev = epochs[i - 1]
        epoch_curr = epochs[i]

        cka_value = float(matrix[i][i - 1])
        local_cka_change = 1.0 - cka_value
        below_tau = int(local_cka_change < tau)

        rows.append({
            "epoch_prev": epoch_prev,
            "epoch_curr": epoch_curr,
            "n_samples": n_samples,
            "cka_value": round(cka_value, 6),
            "local_cka_change": round(local_cka_change, 6),
            "below_tau": below_tau,
        })

    return rows


def derive_pairwise_long_rows(
    epochs: list,
    n_samples: int,
    matrix: np.ndarray,
) -> list:
    """
    Build rows for cka_pairwise_long.csv — all pairs in the upper triangle
    plus the diagonal (epoch_a <= epoch_b).

    The lower triangle is omitted because CKA is symmetric: the value for
    (epoch_a, epoch_b) is the same as (epoch_b, epoch_a).
    """
    rows = []
    n = len(epochs)
    for i in range(n):
        for j in range(i, n):
            rows.append({
                "epoch_a": epochs[i],
                "epoch_b": epochs[j],
                "n_samples": n_samples,
                "cka_value": round(float(matrix[i][j]), 6),
            })
    return rows


def derive_summary_rows(
    epochs: list,
    n_samples: int,
    matrix: np.ndarray,
    tau: float,
) -> list:
    """
    Build rows for cka_summary.csv — one row per checkpoint epoch with four
    summary metrics derived from the full pairwise matrix.

    local_cka_change  = 1 - CKA(t, previous_checkpoint)
                        NaN for the first epoch (no previous exists).
    cka_to_final      = CKA(t, final_epoch)
                        For the final epoch itself this is CKA(final, final) = 1.0.
    mean_future_cka   = mean of CKA(t, t') for all future checkpoints t' > t.
                        NaN for the final epoch (no future checkpoints).
    below_tau         = int(local_cka_change < tau)
                        0 for the first epoch (no measurement available).
    """
    n = len(epochs)
    final_idx = n - 1
    rows = []

    for i in range(n):
        epoch = epochs[i]

        # local_cka_change — compare with the previous checkpoint in the list
        if i == 0:
            local_cka_change = float("nan")
            below_tau = 0
        else:
            cka_prev = float(matrix[i][i - 1])
            local_cka_change = round(1.0 - cka_prev, 6)
            below_tau = int(local_cka_change < tau)

        # cka_to_final — similarity to the last checkpoint
        cka_to_final = round(float(matrix[i][final_idx]), 6)

        # mean_future_cka — average over all later checkpoints
        if i == final_idx:
            mean_future_cka = float("nan")
        else:
            # matrix[i][i+1 : n] contains CKA(epoch_i, epoch_j) for j > i
            future_cka_values = matrix[i][i + 1 : n]
            mean_future_cka = round(float(future_cka_values.mean()), 6)

        rows.append({
            "epoch": epoch,
            "n_samples": n_samples,
            "local_cka_change": local_cka_change,
            "cka_to_final": cka_to_final,
            "mean_future_cka": mean_future_cka,
            "below_tau": below_tau,
        })

    return rows


# ---------------------------------------------------------------------------
# CSV saving
# ---------------------------------------------------------------------------

def save_consecutive_csv(
    rows: list,
    output_path: str,
    logger: logging.Logger,
) -> None:
    fieldnames = [
        "epoch_prev", "epoch_curr", "n_samples",
        "cka_value", "local_cka_change", "below_tau",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved {len(rows)} consecutive pairs  →  {output_path}")


def save_pairwise_long_csv(
    rows: list,
    output_path: str,
    logger: logging.Logger,
) -> None:
    fieldnames = ["epoch_a", "epoch_b", "n_samples", "cka_value"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved {len(rows)} pairwise entries  →  {output_path}")


def save_matrix_csv(
    epochs: list,
    matrix: np.ndarray,
    output_path: str,
    logger: logging.Logger,
) -> None:
    """
    Save the N×N CKA matrix in wide CSV format.

    Header row: 'epoch', followed by each epoch number.
    Data rows:  epoch label in column 0, then N CKA values.

    This format can be loaded with:
        pd.read_csv(path, index_col='epoch')
    to recover the full N×N DataFrame with epoch labels on both axes.
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        header_row = ["epoch"] + [str(e) for e in epochs]
        writer.writerow(header_row)

        for i, epoch in enumerate(epochs):
            row_values = [round(float(v), 6) for v in matrix[i]]
            writer.writerow([epoch] + row_values)

    n = len(epochs)
    logger.info(f"Saved {n}x{n} CKA matrix  →  {output_path}")


def save_summary_csv(
    rows: list,
    output_path: str,
    logger: logging.Logger,
) -> None:
    fieldnames = [
        "epoch", "n_samples",
        "local_cka_change", "cka_to_final", "mean_future_cka", "below_tau",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved summary ({len(rows)} epochs)  →  {output_path}")


# ---------------------------------------------------------------------------
# Verification helpers (run after matrix is computed)
# ---------------------------------------------------------------------------

def verify_matrix_symmetry(matrix: np.ndarray, epochs: list, logger: logging.Logger) -> None:
    """
    Assert that matrix[i][j] == matrix[j][i] for all i, j.
    Raises RuntimeError if symmetry is violated.
    """
    max_asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    if max_asymmetry > 1e-10:
        raise RuntimeError(
            f"CKA matrix symmetry check FAILED: max |K[i,j] - K[j,i]| = {max_asymmetry:.2e}. "
            "This is a bug in compute_pairwise_cka_matrix."
        )
    logger.info(f"Symmetry check passed  (max asymmetry = {max_asymmetry:.2e})")


def verify_diagonal_ones(matrix: np.ndarray, epochs: list, logger: logging.Logger) -> None:
    """
    Assert that matrix[i][i] == 1.0 for all i.
    Raises RuntimeError if any diagonal entry deviates by more than 1e-10.
    """
    diagonal = np.diag(matrix)
    max_diag_error = float(np.max(np.abs(diagonal - 1.0)))
    if max_diag_error > 1e-10:
        raise RuntimeError(
            f"CKA matrix diagonal check FAILED: max |diag - 1| = {max_diag_error:.2e}. "
            "This is a bug in compute_pairwise_cka_matrix."
        )
    logger.info(f"Diagonal check passed  (max |diag - 1| = {max_diag_error:.2e})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Parse arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Compute full temporal CKA trajectory from extracted feature files. "
            "Run features/extract_features.py first."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/resnet18_cifar10_200_fullstudy.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["consecutive", "pairwise", "summary"],
        default="summary",
        help=(
            "consecutive: only cka_consecutive.csv.  "
            "pairwise: consecutive + pairwise_long + matrix CSVs.  "
            "summary (default): all four CSVs including per-epoch summary metrics."
        ),
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default=None,
        help=(
            "Override the cka_subset feature directory. "
            "Default: {config[paths][activations]}/cka_subset"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Override the results output directory from config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override the seed from config. Paths are routed to "
            "seed-specific subdirectories."
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 2. Load config and resolve paths
    # ------------------------------------------------------------------
    config = load_config(args.config)

    if args.seed is not None:
        config["seed"] = args.seed
        for path_key in ["activations", "results", "logs"]:
            if path_key in config["paths"]:
                config["paths"][path_key] = os.path.join(
                    config["paths"][path_key], f"seed_{args.seed}"
                )

    activations_dir = config["paths"]["activations"]
    results_dir = args.results_dir or config["paths"]["results"]
    logs_dir = config["paths"].get("logs", results_dir)

    # CKA subset features live in a subdirectory of activations_dir
    features_dir = args.features_dir or os.path.join(activations_dir, "cka_subset")

    tau = config["tau"]
    seed = config["seed"]

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Logging
    # ------------------------------------------------------------------
    log_path = os.path.join(logs_dir, "cka_trajectory.log")
    logger = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("CKA Trajectory Analysis — Representation Stabilization")
    logger.info("=" * 70)
    logger.info(f"Config:       {args.config}")
    logger.info(f"Mode:         {args.mode}")
    logger.info(f"Features dir: {features_dir}")
    logger.info(f"Results dir:  {results_dir}")
    logger.info(f"tau (reference threshold): {tau}")

    # ------------------------------------------------------------------
    # 4. Reproducibility
    # ------------------------------------------------------------------
    set_seed(seed)
    logger.info(f"Seed: {seed}")

    # ------------------------------------------------------------------
    # 5. Discover feature files
    # ------------------------------------------------------------------
    if not os.path.isdir(features_dir):
        logger.error(
            f"Features directory not found: {features_dir}\n"
            "Run: python features/extract_features.py --config <config> "
            "--sets cka_subset"
        )
        sys.exit(1)

    epochs = find_feature_epochs(features_dir)

    if len(epochs) < 2:
        logger.error(
            f"Need at least 2 feature files to compute CKA. "
            f"Found {len(epochs)} in '{features_dir}'. "
            "Run features/extract_features.py first."
        )
        sys.exit(1)

    logger.info(
        f"Found {len(epochs)} feature files  "
        f"(epochs {epochs[0]} to {epochs[-1]})"
    )
    logger.info(f"Epoch list: {epochs}")

    # ------------------------------------------------------------------
    # 6. Load all features into memory
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("Loading feature files...")
    all_features = load_all_features(features_dir=features_dir, epochs=epochs, logger=logger)

    n_samples = all_features[0][1].shape[0]
    n_features = all_features[0][1].shape[1]
    logger.info(
        f"Loaded {len(all_features)} epochs  "
        f"n_samples={n_samples}  n_features={n_features}"
    )

    # ------------------------------------------------------------------
    # 7. Compute full pairwise CKA matrix
    # ------------------------------------------------------------------
    n_pairs = len(epochs) * (len(epochs) + 1) // 2
    logger.info("-" * 70)
    logger.info(
        f"Computing pairwise CKA matrix  "
        f"({len(epochs)}x{len(epochs)}, {n_pairs} pairs in upper triangle)..."
    )

    cka_matrix = compute_pairwise_cka_matrix(all_features=all_features, logger=logger)

    # ------------------------------------------------------------------
    # 8. Verify matrix properties
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("Verifying matrix properties...")
    verify_matrix_symmetry(cka_matrix, epochs, logger)
    verify_diagonal_ones(cka_matrix, epochs, logger)

    # ------------------------------------------------------------------
    # 9. Derive output rows
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("Deriving output rows...")

    consecutive_rows = derive_consecutive_rows(
        epochs=epochs,
        n_samples=n_samples,
        matrix=cka_matrix,
        tau=tau,
    )
    n_below_tau = sum(r["below_tau"] for r in consecutive_rows)
    logger.info(
        f"  Consecutive pairs: {len(consecutive_rows)}  "
        f"below tau={tau}: {n_below_tau} / {len(consecutive_rows)}"
    )

    pairwise_long_rows = derive_pairwise_long_rows(
        epochs=epochs,
        n_samples=n_samples,
        matrix=cka_matrix,
    )
    logger.info(f"  Pairwise entries (upper triangle + diagonal): {len(pairwise_long_rows)}")

    summary_rows = derive_summary_rows(
        epochs=epochs,
        n_samples=n_samples,
        matrix=cka_matrix,
        tau=tau,
    )
    logger.info(f"  Summary rows: {len(summary_rows)}")

    # ------------------------------------------------------------------
    # 10. Save CSVs based on mode
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info(f"Saving output CSVs (mode={args.mode})...")

    # cka_consecutive.csv — produced by all modes
    consecutive_path = os.path.join(results_dir, "cka_consecutive.csv")
    save_consecutive_csv(
        rows=consecutive_rows,
        output_path=consecutive_path,
        logger=logger,
    )

    # cka_pairwise_long.csv and cka_matrix.csv — pairwise and summary modes
    if args.mode in ("pairwise", "summary"):
        pairwise_long_path = os.path.join(results_dir, "cka_pairwise_long.csv")
        save_pairwise_long_csv(
            rows=pairwise_long_rows,
            output_path=pairwise_long_path,
            logger=logger,
        )

        matrix_path = os.path.join(results_dir, "cka_matrix.csv")
        save_matrix_csv(
            epochs=epochs,
            matrix=cka_matrix,
            output_path=matrix_path,
            logger=logger,
        )

    # cka_summary.csv — summary mode only
    if args.mode == "summary":
        summary_path = os.path.join(results_dir, "cka_summary.csv")
        save_summary_csv(
            rows=summary_rows,
            output_path=summary_path,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # 11. Log summary statistics
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("CKA trajectory analysis complete.")
    logger.info(f"  Epochs processed:   {len(epochs)}")
    logger.info(f"  Samples per epoch:  {n_samples}")
    logger.info(f"  Features per sample: {n_features}")
    logger.info(f"  Consecutive below tau={tau}: {n_below_tau} / {len(consecutive_rows)}")

    # Log the summary table to console for quick inspection
    if args.mode == "summary" and summary_rows:
        logger.info("-" * 70)
        logger.info("Per-epoch summary (first 10 and last 5 rows):")
        logger.info(
            f"  {'epoch':>6}  {'local_change':>13}  {'to_final':>9}  "
            f"{'mean_future':>12}  {'below_tau':>9}"
        )
        logger.info("  " + "-" * 58)

        display_rows = summary_rows[:10] + (summary_rows[-5:] if len(summary_rows) > 10 else [])
        shown_epochs = set()
        for row in display_rows:
            if row["epoch"] in shown_epochs:
                continue
            shown_epochs.add(row["epoch"])

            lc = row["local_cka_change"]
            tf = row["cka_to_final"]
            mf = row["mean_future_cka"]

            lc_str = f"{lc:.4f}" if lc == lc else "nan"
            tf_str = f"{tf:.4f}"
            mf_str = f"{mf:.4f}" if mf == mf else "nan"

            logger.info(
                f"  {row['epoch']:>6}  {lc_str:>13}  {tf_str:>9}  "
                f"{mf_str:>12}  {row['below_tau']:>9}"
            )

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
