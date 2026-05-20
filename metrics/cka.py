"""
metrics/cka.py — Compute temporal CKA between checkpoint epochs.

Reads pre-extracted activation .npy files (produced by train.py) from activations/
and computes linear CKA between epoch pairs.

We measure within-network temporal CKA: CKA(X_t, X_{t-prev}) where t-prev is the
preceding checkpoint epoch. This is NOT cross-network CKA (Kapoor et al. 2025) — we
are measuring how much the same network's representations change over training time.

Reference: Kornblith et al. (2019), "Similarity of Neural Network Representations
Revisited."

Modes:
    consecutive (default) — CKA between each adjacent pair of checkpoints.
                            Output: results/cka_results.csv
    pairwise              — CKA between ALL epoch pairs (full epoch × epoch matrix).
                            Required by analysis/plot_cka_heatmap.py.
                            Output: results/cka_pairwise_results.csv
                            Loads all activation files into memory at once to avoid
                            redundant disk reads.

Usage:
    python metrics/cka.py --config configs/resnet18_cifar10.yaml
    python metrics/cka.py --config configs/resnet18_cifar10.yaml --mode pairwise
"""

import argparse
import csv
import logging
import os
import sys

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Setup utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("cka")
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
# Activation file discovery
# ---------------------------------------------------------------------------

def find_activation_files(activations_dir: str) -> list[tuple[int, str]]:
    """
    Discover all activations_epoch_XXXX.npy files in activations_dir.
    Returns a list of (epoch, filepath) pairs sorted by epoch ascending.
    Filename convention is defined by train.py: activations_epoch_{epoch:04d}.npy
    """
    epoch_path_pairs = []

    for filename in os.listdir(activations_dir):
        if not filename.startswith("activations_epoch_"):
            continue
        if not filename.endswith(".npy"):
            continue

        epoch_str = filename.replace("activations_epoch_", "").replace(".npy", "")
        epoch = int(epoch_str)
        full_path = os.path.join(activations_dir, filename)
        epoch_path_pairs.append((epoch, full_path))

    epoch_path_pairs.sort(key=lambda pair: pair[0])
    return epoch_path_pairs


# ---------------------------------------------------------------------------
# CKA computation (Kornblith et al. 2019, linear kernel)
# ---------------------------------------------------------------------------

def center_gram_matrix(gram_matrix: np.ndarray) -> np.ndarray:
    """
    Double-center a Gram matrix: K_c = H K H where H = I - (1/n) 1 1^T.

    Equivalent to subtracting each row mean, each column mean, and adding the
    grand mean back. This centering makes CKA invariant to additive constant
    offsets in the representation (Kornblith et al. 2019, Appendix A).
    """
    row_means = gram_matrix.mean(axis=1, keepdims=True)   # shape (n, 1)
    col_means = gram_matrix.mean(axis=0, keepdims=True)   # shape (1, n)
    grand_mean = gram_matrix.mean()                        # scalar

    gram_matrix_centered = gram_matrix - row_means - col_means + grand_mean
    return gram_matrix_centered


def compute_hsic(K_centered: np.ndarray, L_centered: np.ndarray) -> float:
    """
    Biased HSIC from pre-centered Gram matrices.

    HSIC(K, L) = (1 / (n-1)^2) * trace(K_c @ L_c)

    We compute the Frobenius inner product instead of materializing K_c @ L_c:
        trace(K_c @ L_c) = sum_{ij} K_c[i,j] * L_c[i,j]
    This avoids an O(n^3) matmul at the cost of O(n^2) memory, which is already
    required to hold both Gram matrices.

    Note on estimator choice: the biased HSIC estimator has a bias of O(1/n).
    For n=2040 this bias is negligible. The unbiased estimator (Song et al. 2012)
    would be more correct for n < ~100, but adds implementation complexity without
    meaningful benefit at our sample size.
    """
    n_samples = K_centered.shape[0]
    normalization = float((n_samples - 1) ** 2)

    frobenius_inner_product = float(np.sum(K_centered * L_centered))
    hsic_value = frobenius_inner_product / normalization
    return hsic_value


def compute_cka(
    activations_prev: np.ndarray,
    activations_curr: np.ndarray,
) -> float:
    """
    Linear CKA between activations from two consecutive checkpoint epochs.

    activations_prev: (n_samples, n_features) at epoch t-prev
    activations_curr: (n_samples, n_features) at epoch t

    CKA = HSIC(K, L) / sqrt(HSIC(K,K) * HSIC(L,L))
    where K = X X^T (linear Gram matrix for X = activations_prev)
    and   L = Y Y^T (linear Gram matrix for Y = activations_curr)

    Returns a float in [0, 1].
    CKA = 1 means identical representations up to orthogonal transformation and
    isotropic scaling. CKA near 0 means geometrically very different representations.

    The stabilization criterion uses (1 - CKA) as the change metric compared to tau.
    """
    gram_prev = activations_prev @ activations_prev.T   # shape (n, n)
    gram_curr = activations_curr @ activations_curr.T   # shape (n, n)

    gram_prev_centered = center_gram_matrix(gram_prev)
    gram_curr_centered = center_gram_matrix(gram_curr)

    hsic_prev_curr = compute_hsic(gram_prev_centered, gram_curr_centered)
    hsic_prev_prev = compute_hsic(gram_prev_centered, gram_prev_centered)
    hsic_curr_curr = compute_hsic(gram_curr_centered, gram_curr_centered)

    denominator = np.sqrt(hsic_prev_prev * hsic_curr_curr)

    if denominator < 1e-10:
        # Both representations have near-zero variance. This can happen at very
        # early epochs if the network outputs near-constant activations (e.g.,
        # before the first meaningful gradient update). Return 0.0 rather than NaN.
        return 0.0

    cka_value = hsic_prev_curr / denominator
    return float(cka_value)


# ---------------------------------------------------------------------------
# Main computation loops
# ---------------------------------------------------------------------------

def run_cka_computation(
    activations_dir: str,
    results_dir: str,
    config: dict,
    logger: logging.Logger,
) -> None:
    """
    Load all activation files and compute CKA for every consecutive epoch pair.
    Saves results to results/cka_results.csv.
    """
    tau = config["tau"]
    logger.info(f"Stabilization threshold tau = {tau}")

    epoch_path_pairs = find_activation_files(activations_dir)
    n_files = len(epoch_path_pairs)
    logger.info(f"Found {n_files} activation files in '{activations_dir}'")

    if n_files < 2:
        logger.error(
            f"Need at least 2 activation files to compute CKA. "
            f"Found only {n_files}. Run train.py first."
        )
        return

    epochs_found = [epoch for epoch, _ in epoch_path_pairs]
    logger.info(f"Epoch range: {epochs_found[0]} to {epochs_found[-1]}")
    logger.info(f"Number of CKA pairs to compute: {n_files - 1}")
    logger.info("-" * 70)

    result_rows = []

    for pair_idx in range(1, n_files):
        epoch_prev, path_prev = epoch_path_pairs[pair_idx - 1]
        epoch_curr, path_curr = epoch_path_pairs[pair_idx]

        activations_prev = np.load(path_prev)   # (n_samples, n_features)
        activations_curr = np.load(path_curr)   # (n_samples, n_features)

        n_samples = activations_prev.shape[0]

        cka_value = compute_cka(activations_prev, activations_curr)

        # (1 - CKA) is the change metric compared against tau.
        # Low change (<tau) means representations are similar — potential stabilization.
        cka_change = 1.0 - cka_value
        is_below_tau = cka_change < tau

        logger.info(
            f"  Pair {pair_idx:3d}/{n_files - 1}"
            f"  epoch {epoch_prev:4d} → {epoch_curr:4d}"
            f"  CKA = {cka_value:.4f}"
            f"  change = {cka_change:.4f}"
            f"  below_tau = {'YES' if is_below_tau else 'no '}"
        )

        result_rows.append({
            "epoch_prev": epoch_prev,
            "epoch_curr": epoch_curr,
            "n_samples": n_samples,
            "cka_value": round(cka_value, 6),
            "cka_change": round(cka_change, 6),
            "below_tau": int(is_below_tau),
        })

    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "cka_results.csv")

    fieldnames = ["epoch_prev", "epoch_curr", "n_samples", "cka_value", "cka_change", "below_tau"]
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    logger.info("-" * 70)
    logger.info(f"Saved CKA results ({len(result_rows)} pairs) to: {output_path}")
    logger.info(
        f"Pairs below tau={tau}: "
        f"{sum(r['below_tau'] for r in result_rows)} / {len(result_rows)}"
    )


def run_cka_pairwise_computation(
    activations_dir: str,
    results_dir: str,
    config: dict,
    logger: logging.Logger,
) -> None:
    """
    Load all activation files and compute CKA for ALL epoch pairs.
    Produces the full epoch × epoch matrix required by plot_cka_heatmap.py.

    Stores only the upper triangle (epoch_a <= epoch_b) because CKA is symmetric.
    The diagonal (epoch_a == epoch_b) is always 1.0 by definition and is included
    so that plot_cka_heatmap.py can reconstruct the complete n × n matrix.

    Memory note: all activation files are loaded once into RAM before the pair loop
    to avoid O(n²) redundant disk reads. For 40 checkpoints × 2040 × 512 float32
    this is ~170 MB — well within the cluster's available memory.
    """
    epoch_path_pairs = find_activation_files(activations_dir)
    n_files          = len(epoch_path_pairs)

    if n_files < 2:
        logger.error(
            f"Need at least 2 activation files to compute CKA. "
            f"Found only {n_files}. Run train.py first."
        )
        return

    n_pairs_upper_triangle = n_files * (n_files + 1) // 2
    logger.info(f"Found {n_files} activation files.")
    logger.info(f"Computing {n_pairs_upper_triangle} pairs (upper triangle + diagonal).")

    # Load all activations into memory once
    logger.info("Loading all activation files into memory...")
    all_activations = []
    for epoch, path in epoch_path_pairs:
        activations = np.load(path)   # shape (n_samples, n_features)
        all_activations.append((epoch, activations))
        logger.info(f"  Loaded epoch {epoch:4d}  shape = {activations.shape}")

    result_rows = []
    pair_count  = 0

    for i in range(n_files):
        epoch_a, activations_a = all_activations[i]

        for j in range(i, n_files):
            epoch_b, activations_b = all_activations[j]

            if i == j:
                # Diagonal: a representation is always identical to itself
                cka_value = 1.0
            else:
                cka_value = compute_cka(activations_a, activations_b)

            pair_count += 1
            logger.info(
                f"  Pair {pair_count:4d}/{n_pairs_upper_triangle}"
                f"  ({epoch_a:4d}, {epoch_b:4d})"
                f"  CKA = {cka_value:.4f}"
            )

            result_rows.append({
                "epoch_a":   epoch_a,
                "epoch_b":   epoch_b,
                "cka_value": round(cka_value, 6),
            })

    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "cka_pairwise_results.csv")

    fieldnames = ["epoch_a", "epoch_b", "cka_value"]
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    logger.info("-" * 70)
    logger.info(f"Saved {len(result_rows)} pairwise CKA values to: {output_path}")
    logger.info(
        f"Use 'python analysis/plot_cka_heatmap.py' to visualise the matrix."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute temporal linear CKA between checkpoint epochs. "
            "Run train.py first to generate activation .npy files."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/resnet18_cifar10.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--activations-dir",
        type=str,
        default=None,
        help="Override the activations directory from config.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Override the results directory from config.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["consecutive", "pairwise"],
        default="consecutive",
        help=(
            "consecutive (default): CKA between adjacent checkpoint pairs only. "
            "pairwise: CKA between ALL epoch pairs — required for the heatmap."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override the seed from config. Input/output paths are routed to "
            "seed-specific subdirectories (e.g. activations/seed_42/)."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # --seed overrides the random seed and routes all paths to a
    # seed-specific subdirectory so concurrent seed runs don't collide.
    if args.seed is not None:
        config["seed"] = args.seed
        config["paths"]["checkpoints"] = os.path.join(
            config["paths"]["checkpoints"], f"seed_{args.seed}"
        )
        config["paths"]["activations"] = os.path.join(
            config["paths"]["activations"], f"seed_{args.seed}"
        )
        config["paths"]["results"] = os.path.join(
            config["paths"]["results"], f"seed_{args.seed}"
        )

    activations_dir = args.activations_dir or config["paths"]["activations"]
    results_dir = args.results_dir or config["paths"]["results"]

    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "cka.log")
    logger = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("CKA Computation — Representation Stabilization")
    logger.info("=" * 70)
    logger.info(f"Config:          {args.config}")
    logger.info(f"Activations dir: {activations_dir}")
    logger.info(f"Results dir:     {results_dir}")

    seed = config["seed"]
    set_seed(seed)
    logger.info(f"Seed: {seed}")
    logger.info(f"Mode: {args.mode}")

    if args.mode == "consecutive":
        run_cka_computation(
            activations_dir=activations_dir,
            results_dir=results_dir,
            config=config,
            logger=logger,
        )
    else:
        run_cka_pairwise_computation(
            activations_dir=activations_dir,
            results_dir=results_dir,
            config=config,
            logger=logger,
        )

    logger.info("=" * 70)
    logger.info("CKA computation complete.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
