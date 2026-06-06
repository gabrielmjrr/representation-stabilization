"""
metrics/entk_trajectory.py — Bounded last-layer empirical NTK trajectory.

IMPORTANT — what this is and is not
-------------------------------------
This is NOT the full-parameter NTK.  The full-parameter NTK requires
forward-mode autodiff (Jacobian-vector products) over all model parameters
and is orders of magnitude more expensive.

This is the LAST-LAYER empirical NTK, computed in closed form from the
pre-extracted penultimate features and the saved linear classifier head.
It is a lightweight training-dynamics diagnostic comparable to CKA.

Closed-form derivation
-----------------------
For a final linear classifier with weight W ∈ R^{C×p} and bias b ∈ R^C:

    z_c(x) = w_c^T h(x) + b_c           [score for class c]
    f_i     = z_{y_i}(x_i)              [score at the true class for example i]

The gradient of f_i with respect to the classifier parameters is:

    ∂f_i/∂w_c = h_i   if c == y_i,  else 0   (only column y_i is non-zero)
    ∂f_i/∂b_c = 1     if c == y_i,  else 0

The NTK entry K[i,j] = <∇_{W,b} f_i, ∇_{W,b} f_j> is therefore:

    K[i,j] = 0                               if y_i ≠ y_j
    K[i,j] = h_i^T h_j  +  bias_flag        if y_i == y_j

where bias_flag = 1 if the FC layer has a bias, 0 otherwise.

Cross-class entries are exactly zero by construction — this is structural,
not numerical.  The within-class/between-class ratio metric reflects this.

Subset strategy
----------------
A fixed stratified subset of n_subset training examples (default 500 = 50
per class for CIFAR-10) is selected once, saved to entk_subset_indices.json,
and reused across all checkpoints in a run.  If the JSON already exists,
it is loaded without resampling, guaranteeing reproducibility across partial
reruns.

Feature loading
----------------
Features are read from full_train/features_epoch_XXXX.npy (the same files
used by neural_collapse.py).  The full feature matrix is loaded, the subset
rows are extracted, and the full matrix is released before computing the
kernel.  Kernels for all epochs are kept in memory simultaneously for the
second pass that computes trajectory metrics.

Memory estimate (n_subset=500, 45 epochs):
  per kernel:   500 × 500 × 8 bytes = 2 MB
  all kernels:  45 × 2 MB = 90 MB
  per feature file: 50 000 × 512 × 4 bytes ≈ 100 MB (loaded then released)
  peak usage: ≈ 190 MB — well within cluster RAM.

Metrics computed
-----------------
  entk_distance_prev
      Kernel distance to the previous checkpoint:
          1 − <K_t, K_{t-1}>_F / (‖K_t‖_F ‖K_{t-1}‖_F)
      NaN for the first checkpoint.

  entk_distance_final
      Kernel distance to the final checkpoint K_T:
          1 − <K_t, K_T>_F / (‖K_t‖_F ‖K_T‖_F)

  mean_future_entk_similarity
      Mean cosine similarity to all future kernels K_s with s > t:
          mean_{s>t}  <K_t, K_s>_F / (‖K_t‖_F ‖K_s‖_F)
      NaN for the final checkpoint.  Mirrors mean_future_cka.

  entk_within_class_mean
      Mean K[i,j] over off-diagonal same-class pairs in the subset.

  entk_between_class_mean
      Mean K[i,j] over all cross-class pairs (= 0 always for last-layer NTK).

  entk_within_between_ratio
      entk_within_class_mean / max(|entk_between_class_mean|, eps),
      eps = 1e-12.

Inputs (under config["paths"]["activations"])
  full_train/labels.npy               (N,)      — static; loaded once
  full_train/features_epoch_XXXX.npy  (N, p)    — loaded per epoch, subset extracted

Inputs (under config["paths"]["checkpoints"])
  checkpoint_epoch_XXXX.pt            — used only to detect presence of fc.bias

Outputs (under config["paths"]["results"])
  entk_summary.csv           — one row per checkpoint epoch
  entk_subset_indices.json   — fixed stratified training-set indices

Optional (if entk.save_matrices: true; stored under paths.base/entk)
  entk_matrix_epoch_XXXX.npy  — raw float32 kernel per epoch (heavy artifact)

Usage
-----
  python metrics/entk_trajectory.py \\
      --config configs/resnet18_cifar10_200_fullstudy.yaml

  python metrics/entk_trajectory.py --config ... --epochs 0 50 100 200
"""

import argparse
import csv
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Reproducibility, logging, config
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("entk_trajectory")
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
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Feature file discovery
# ---------------------------------------------------------------------------

def find_feature_epochs(full_train_dir: str) -> list:
    """
    Discover available checkpoint epochs by listing features_epoch_XXXX.npy
    files in the full_train directory.  Returns a numerically sorted list.
    """
    epochs = []
    for filename in os.listdir(full_train_dir):
        if not filename.startswith("features_epoch_"):
            continue
        if not filename.endswith(".npy"):
            continue
        epoch_str = filename.replace("features_epoch_", "").replace(".npy", "")
        epochs.append(int(epoch_str))
    epochs.sort()
    return epochs


def validate_feature_epochs(
    found_epochs: list,
    expected_epochs: list,
    logger: logging.Logger,
) -> None:
    """
    Fullstudy runs should process exactly the configured checkpoint epochs.
    Manual --epochs is the explicit partial/debug path.
    """
    found_sorted = sorted(found_epochs)
    expected_sorted = sorted(expected_epochs)

    if found_sorted == expected_sorted:
        logger.info(f"Epoch completeness check passed ({len(found_sorted)} epochs).")
        return

    missing = sorted(set(expected_sorted) - set(found_sorted))
    extra = sorted(set(found_sorted) - set(expected_sorted))
    logger.error(
        "Feature epoch completeness check FAILED.\n"
        f"  Expected epochs: {expected_sorted}\n"
        f"  Found epochs:    {found_sorted}\n"
        f"  Missing epochs:  {missing}\n"
        f"  Extra epochs:    {extra}\n"
        "Run features/extract_features.py for all checkpoint epochs, or pass "
        "--epochs explicitly for a partial smoke test."
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Label and feature loading
# ---------------------------------------------------------------------------

def load_train_labels(full_train_dir: str, logger: logging.Logger) -> np.ndarray:
    """
    Load the static training-set label array from full_train/labels.npy.
    Labels do not change across epochs; load them once and reuse.
    """
    labels_path = os.path.join(full_train_dir, "labels.npy")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Training labels not found: {labels_path}\n"
            "Run:  python features/extract_features.py --sets full_train"
        )
    labels = np.load(labels_path).astype(np.int64)
    logger.info(f"Labels loaded: shape={labels.shape}  path={labels_path}")
    return labels


def load_full_features(epoch: int, full_train_dir: str, logger: logging.Logger) -> np.ndarray:
    """
    Load the full (N, p) feature array for one checkpoint epoch.
    The caller extracts the eNTK subset and should delete the full array
    afterwards to release memory.
    """
    feature_path = os.path.join(full_train_dir, f"features_epoch_{epoch:04d}.npy")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(
            f"Feature file not found for epoch {epoch}: {feature_path}\n"
            "Run:  python features/extract_features.py --sets full_train"
        )
    features = np.load(feature_path)
    logger.info(
        f"  Loaded full_train features epoch {epoch:4d}: "
        f"shape={features.shape}  path={feature_path}"
    )
    return features


def detect_classifier_bias(
    epoch: int,
    checkpoint_dir: str,
    logger: logging.Logger,
) -> bool:
    """
    Load checkpoint for the given epoch and return True if fc.bias is present.
    Only the state dict is accessed; weights are not used in kernel computation.
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found for epoch {epoch}: {checkpoint_path}\n"
            "Run train.py first."
        )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = ckpt["model_state_dict"]
    has_bias = ("fc.bias" in state) and (state["fc.bias"] is not None)
    logger.info(f"  Checkpoint epoch {epoch}: fc.bias present = {has_bias}")
    return has_bias


# ---------------------------------------------------------------------------
# Stratified subset management
# ---------------------------------------------------------------------------

def compute_stratified_subset_indices(
    labels: np.ndarray,
    n_subset: int,
    n_classes: int,
    seed: int,
    logger: logging.Logger,
) -> list:
    """
    Compute a balanced stratified subset of n_subset training indices.

    n_per_class = n_subset // n_classes examples are drawn per class using
    numpy RandomState(seed), so the result is deterministic.

    Raises ValueError if n_subset is too small or any class lacks enough examples.
    Returns a sorted list of int indices into the full training set.
    """
    n_per_class = n_subset // n_classes
    if n_per_class == 0:
        raise ValueError(
            f"n_subset={n_subset} too small for n_classes={n_classes}: "
            f"n_subset // n_classes = 0.  Increase entk.n_subset."
        )

    rng = np.random.RandomState(seed)
    selected = []

    for c in range(n_classes):
        class_indices = np.where(labels == c)[0]
        if len(class_indices) < n_per_class:
            raise ValueError(
                f"Class {c} has only {len(class_indices)} training examples "
                f"but n_per_class={n_per_class} requested.  "
                f"Reduce entk.n_subset or check the training-set labels."
            )
        sampled = rng.choice(class_indices, size=n_per_class, replace=False)
        selected.extend(sampled.tolist())

    selected.sort()
    logger.info(
        f"Stratified subset: {len(selected)} indices "
        f"({n_per_class} per class × {n_classes} classes)"
    )
    return selected


def load_or_create_subset_indices(
    results_dir: str,
    labels: np.ndarray,
    n_subset: int,
    n_classes: int,
    seed: int,
    logger: logging.Logger,
) -> list:
    """
    Load entk_subset_indices.json if it exists (reuse without resampling),
    or compute a new stratified subset and save it.

    The JSON file is text-based so it is human-readable and can be committed
    to git for reproducibility.
    """
    indices_path = os.path.join(results_dir, "entk_subset_indices.json")

    if os.path.exists(indices_path):
        with open(indices_path, "r") as f:
            indices = json.load(f)
        logger.info(
            f"Loaded existing eNTK subset ({len(indices)} indices): {indices_path}"
        )
        if len(indices) == 0:
            raise ValueError(f"entk_subset_indices.json is empty: {indices_path}")
        max_idx = max(indices)
        if max_idx >= len(labels):
            raise ValueError(
                f"Subset index {max_idx} is out of range for training set "
                f"of length {len(labels)}.  Delete {indices_path} and rerun."
            )
        return indices

    indices = compute_stratified_subset_indices(
        labels=labels,
        n_subset=n_subset,
        n_classes=n_classes,
        seed=seed,
        logger=logger,
    )
    with open(indices_path, "w") as f:
        json.dump(indices, f)
    logger.info(f"Saved eNTK subset indices ({len(indices)}): {indices_path}")
    return indices


# ---------------------------------------------------------------------------
# Last-layer eNTK — closed form, no autograd
# ---------------------------------------------------------------------------

def compute_last_layer_entk(
    H_sub: np.ndarray,
    y_sub: np.ndarray,
    has_bias: bool,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Compute the last-layer empirical NTK for the selected training subset.

    Formula:
        K[i,j] = 1[y_i == y_j] * (h_i^T h_j + bias_flag)

    where bias_flag = 1 if the FC layer has a bias term, 0 otherwise.

    Arguments:
        H_sub:    (n_subset, p) float64 — penultimate features for subset
        y_sub:    (n_subset,) int64     — class labels for subset examples
        has_bias: bool                  — whether the FC layer has a bias term

    Returns:
        K: (n_subset, n_subset) float64, symmetric
    """
    H = H_sub.astype(np.float64)               # (n_subset, p)

    # h_i^T h_j for all pairs: (n_subset, n_subset)
    gram = H @ H.T

    # Binary class-match mask: same_class[i,j] = 1 if y_i == y_j
    same_class = (y_sub[:, None] == y_sub[None, :]).astype(np.float64)

    bias_add = 1.0 if has_bias else 0.0

    # K[i,j] = 1[y_i==y_j] * (h_i^T h_j + bias_flag)
    # Cross-class entries are exactly zero.
    K = same_class * (gram + bias_add)

    frob_norm = float(np.linalg.norm(K, "fro"))
    diag_mean = float(np.diag(K).mean())
    logger.info(
        f"  eNTK kernel: shape={K.shape}  "
        f"bias_included={has_bias}  "
        f"‖K‖_F={frob_norm:.4f}  "
        f"diag_mean={diag_mean:.4f}"
    )
    return K


# ---------------------------------------------------------------------------
# Kernel similarity and distance utilities
# ---------------------------------------------------------------------------

def kernel_cosine_similarity(
    K_a: np.ndarray,
    K_b: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Cosine similarity between two kernel matrices treated as flat vectors:
        sim = <K_a, K_b>_F / (‖K_a‖_F × ‖K_b‖_F)

    Equivalent to the normalized Frobenius inner product.
    Returns 0.0 if either kernel has near-zero Frobenius norm.
    """
    inner  = float(np.sum(K_a * K_b))
    norm_a = float(np.sqrt(np.sum(K_a ** 2)))
    norm_b = float(np.sqrt(np.sum(K_b ** 2)))
    denom  = norm_a * norm_b
    if denom < eps:
        return 0.0
    return inner / denom


def kernel_distance(
    K_a: np.ndarray,
    K_b: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Kernel distance: 1 − cosine_similarity(K_a, K_b).
    Ranges in [0, 2].  Equals 0 when kernels are proportional.
    """
    return 1.0 - kernel_cosine_similarity(K_a, K_b, eps)


def compute_within_between_ratio(
    K: np.ndarray,
    y_sub: np.ndarray,
    eps: float = 1e-12,
) -> tuple:
    """
    Compute within-class and between-class kernel means and their ratio.

    Within-class mean:  mean K[i,j] over off-diagonal same-class pairs.
    Between-class mean: mean K[i,j] over all cross-class pairs.

    Note: for the last-layer eNTK, all cross-class entries are exactly zero,
    so between_mean = 0.0 always and ratio = within_mean / eps (large).
    Both raw means are returned so they can be interpreted independently.

    Returns:
        (within_mean, between_mean, ratio)
    """
    n = len(y_sub)
    same_class = (y_sub[:, None] == y_sub[None, :])
    diagonal   = np.eye(n, dtype=bool)

    within_mask  = same_class & ~diagonal
    between_mask = ~same_class

    within_vals  = K[within_mask]
    between_vals = K[between_mask]

    within_mean  = float(within_vals.mean())  if len(within_vals)  > 0 else float("nan")
    between_mean = float(between_vals.mean()) if len(between_vals) > 0 else float("nan")

    if np.isnan(within_mean) or np.isnan(between_mean):
        ratio = float("nan")
    else:
        ratio = within_mean / max(abs(between_mean), eps)

    return within_mean, between_mean, ratio


# ---------------------------------------------------------------------------
# CSV saving
# ---------------------------------------------------------------------------

def save_entk_summary_csv(
    rows: list,
    output_path: str,
    logger: logging.Logger,
) -> None:
    """Save entk_summary.csv — one row per checkpoint epoch."""
    fieldnames = [
        "epoch",
        "entk_distance_prev",
        "entk_distance_final",
        "mean_future_entk_similarity",
        "entk_within_class_mean",
        "entk_between_class_mean",
        "entk_within_between_ratio",
        "n_subset",
        "per_class_subset_size",
        "bias_included",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved entk_summary.csv ({len(rows)} rows)  ->  {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Parse arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Compute last-layer eNTK trajectory from pre-extracted features. "
            "NOT full-parameter NTK — uses closed-form classifier-gradient kernel. "
            "Run features/extract_features.py --sets full_train and train.py first."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/resnet18_cifar10_200_fullstudy.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Specific epochs to process (e.g. --epochs 0 50 100 200). "
            "Default: all epochs found in full_train feature directory."
        ),
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default=None,
        help="Override the activations base directory from config.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default=None,
        help="Override the checkpoints directory from config.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Override the results directory from config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the global random seed from config.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 2. Load config and resolve paths
    # ------------------------------------------------------------------
    config = load_config(args.config)

    if args.seed is not None:
        config["seed"] = args.seed

    seed            = config["seed"]
    activations_dir = args.features_dir   or config["paths"]["activations"]
    checkpoint_dir  = args.checkpoints_dir or config["paths"]["checkpoints"]
    results_dir     = args.results_dir    or config["paths"]["results"]
    logs_dir        = config["paths"].get("logs", results_dir)
    full_train_dir  = os.path.join(activations_dir, "full_train")

    # eNTK-specific config (all keys optional with defaults)
    entk_cfg    = config.get("entk", {})
    enabled     = entk_cfg.get("enabled", True)
    n_subset    = int(entk_cfg.get("n_subset", 500))
    entk_seed   = int(entk_cfg.get("seed", seed))
    save_mats   = bool(entk_cfg.get("save_matrices", False))
    # Matrix output dir: config key, or fall back to paths.base/entk
    mat_out_dir = entk_cfg.get("output_dir") or os.path.join(
        config["paths"].get("base", results_dir), "entk"
    )

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    if save_mats:
        os.makedirs(mat_out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Logging
    # ------------------------------------------------------------------
    log_path = os.path.join(logs_dir, "entk_trajectory.log")
    logger = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("Last-Layer eNTK Trajectory (closed-form, no autograd)")
    logger.info("=" * 70)
    logger.info(f"Config:          {args.config}")
    logger.info(f"Activations dir: {activations_dir}")
    logger.info(f"Checkpoint dir:  {checkpoint_dir}")
    logger.info(f"Results dir:     {results_dir}")
    logger.info(f"Global seed:     {seed}  eNTK subset seed: {entk_seed}")
    logger.info(f"n_subset:        {n_subset}")
    logger.info(f"save_matrices:   {save_mats}")
    if save_mats:
        logger.info(f"Matrix out dir:  {mat_out_dir}")

    if not enabled:
        logger.info("entk.enabled=false in config — exiting without computing.")
        return

    # ------------------------------------------------------------------
    # 4. Reproducibility
    # ------------------------------------------------------------------
    set_seed(seed)

    # ------------------------------------------------------------------
    # 5. Discover feature epochs
    # ------------------------------------------------------------------
    if not os.path.isdir(full_train_dir):
        logger.error(
            f"full_train directory not found: {full_train_dir}\n"
            "Run:  python features/extract_features.py --sets full_train"
        )
        sys.exit(1)

    discovered_epochs = find_feature_epochs(full_train_dir)

    if args.epochs is not None:
        epochs_to_process = sorted(args.epochs)
        logger.info(f"Epochs (from --epochs): {epochs_to_process}")
    else:
        validate_feature_epochs(
            found_epochs=discovered_epochs,
            expected_epochs=config["checkpoint_epochs"],
            logger=logger,
        )
        epochs_to_process = discovered_epochs
        logger.info(
            f"Epochs discovered: {len(epochs_to_process)} "
            f"(range {epochs_to_process[0]}–{epochs_to_process[-1]})"
        )

    if len(epochs_to_process) < 2:
        logger.error(
            f"Need at least 2 epochs for trajectory metrics, "
            f"found {len(epochs_to_process)}."
        )
        sys.exit(1)

    logger.info(f"Epoch list: {epochs_to_process}")

    # ------------------------------------------------------------------
    # 6. Load training labels (static across epochs)
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("Loading training labels (static across epochs)...")
    train_labels = load_train_labels(full_train_dir, logger)

    n_train        = len(train_labels)
    unique_classes = np.unique(train_labels)
    n_classes      = len(unique_classes)

    logger.info(f"Training examples: {n_train}  Classes: {n_classes}")
    logger.info(f"Class labels: {unique_classes.tolist()}")

    if sorted(unique_classes.tolist()) != list(range(n_classes)):
        logger.error(
            f"Expected class labels 0..{n_classes - 1}, "
            f"found {unique_classes.tolist()}."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 7. Load or create stratified subset indices
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("Resolving eNTK subset indices...")

    try:
        subset_indices = load_or_create_subset_indices(
            results_dir=results_dir,
            labels=train_labels,
            n_subset=n_subset,
            n_classes=n_classes,
            seed=entk_seed,
            logger=logger,
        )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    n_actual   = len(subset_indices)
    n_per_cls  = n_actual // n_classes
    y_sub      = train_labels[subset_indices]        # (n_actual,) — static

    # Verify balance
    per_class_counts = {int(c): int((y_sub == c).sum()) for c in unique_classes}
    logger.info(f"Subset per-class counts: {per_class_counts}")
    uneven = [c for c, cnt in per_class_counts.items() if cnt != n_per_cls]
    if uneven:
        logger.warning(
            f"Subset is not perfectly balanced for classes {uneven}."
        )

    # Log kernel memory estimate
    kernel_bytes = n_actual * n_actual * 8
    total_bytes  = len(epochs_to_process) * kernel_bytes
    logger.info(
        f"Kernel shape: ({n_actual}, {n_actual})  "
        f"per kernel: {kernel_bytes / 1e6:.1f} MB  "
        f"all kernels in memory: {total_bytes / 1e6:.1f} MB"
    )

    # ------------------------------------------------------------------
    # 8. Detect classifier bias from the first checkpoint
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    first_epoch = epochs_to_process[0]
    logger.info(f"Detecting fc.bias presence from checkpoint epoch {first_epoch}...")

    try:
        has_bias = detect_classifier_bias(first_epoch, checkpoint_dir, logger)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Warn if the final checkpoint is missing
    final_epoch      = epochs_to_process[-1]
    final_ckpt_path  = os.path.join(
        checkpoint_dir, f"checkpoint_epoch_{final_epoch:04d}.pt"
    )
    if not os.path.exists(final_ckpt_path):
        logger.warning(
            f"Final checkpoint not found: {final_ckpt_path}.  "
            "entk_distance_final is relative to the last successfully "
            "computed kernel, which may not be the end of training."
        )

    # ------------------------------------------------------------------
    # 9. First pass — compute kernels for all epochs
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info(
        f"Pass 1: computing eNTK kernels for {len(epochs_to_process)} epochs..."
    )
    logger.info("=" * 70)

    all_kernels = []   # list of (epoch: int, K: np.ndarray)

    for epoch_idx, epoch in enumerate(epochs_to_process):
        t0 = time.perf_counter()
        logger.info(f"Epoch {epoch:4d}  [{epoch_idx + 1}/{len(epochs_to_process)}]")

        # Load full feature matrix
        try:
            H_full = load_full_features(epoch, full_train_dir, logger)
        except FileNotFoundError as e:
            logger.error(f"  Skipping epoch {epoch}: {e}")
            continue

        # Validate shape
        if H_full.ndim != 2:
            logger.error(
                f"  Epoch {epoch}: features must be 2D [N, p], "
                f"got shape {H_full.shape}. Skipping."
            )
            continue
        if H_full.shape[0] != n_train:
            logger.error(
                f"  Epoch {epoch}: expected {n_train} rows, "
                f"got {H_full.shape[0]}. Skipping."
            )
            continue

        # Extract subset and release full matrix
        H_sub  = H_full[subset_indices]   # (n_actual, p)
        p      = H_sub.shape[1]
        del H_full

        logger.info(
            f"  H_sub: shape={H_sub.shape}  dtype={H_sub.dtype}  "
            f"mean={H_sub.mean():.4f}  std={H_sub.std():.4f}"
        )

        # Closed-form last-layer eNTK
        K_t = compute_last_layer_entk(
            H_sub=H_sub,
            y_sub=y_sub,
            has_bias=has_bias,
            logger=logger,
        )
        all_kernels.append((epoch, K_t))

        # Optional: save kernel matrix as a heavy artifact
        if save_mats:
            mat_path = os.path.join(
                mat_out_dir, f"entk_matrix_epoch_{epoch:04d}.npy"
            )
            np.save(mat_path, K_t.astype(np.float32))
            logger.info(f"  Saved kernel: {mat_path}")

        elapsed = time.perf_counter() - t0
        logger.info(f"  Epoch {epoch} done in {elapsed:.2f}s")

    n_computed = len(all_kernels)
    n_skipped  = len(epochs_to_process) - n_computed

    logger.info(
        f"Pass 1 complete: {n_computed} kernels computed, {n_skipped} skipped."
    )

    if n_computed < 2:
        logger.error(
            f"Only {n_computed} kernel(s) computed successfully. "
            "Need at least 2 for trajectory metrics.  "
            "Check feature files and logs above."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 10. Second pass — compute all trajectory metrics
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("Pass 2: computing trajectory metrics from stored kernels...")
    logger.info("=" * 70)

    final_K = all_kernels[-1][1]
    all_result_rows = []

    for i, (epoch, K_t) in enumerate(all_kernels):

        # entk_distance_prev
        if i == 0:
            dist_prev = float("nan")
        else:
            dist_prev = kernel_distance(K_t, all_kernels[i - 1][1])

        # entk_distance_final
        dist_final = kernel_distance(K_t, final_K)

        # mean_future_entk_similarity
        if i == n_computed - 1:
            mean_future_sim = float("nan")
        else:
            future_sims = [
                kernel_cosine_similarity(K_t, all_kernels[j][1])
                for j in range(i + 1, n_computed)
            ]
            mean_future_sim = float(np.mean(future_sims))

        # Within/between kernel means and ratio
        within_mean, between_mean, ratio = compute_within_between_ratio(
            K=K_t, y_sub=y_sub
        )

        def _r(v: float, n: int = 8) -> float:
            return float("nan") if np.isnan(v) else round(v, n)

        result_row = {
            "epoch":                       epoch,
            "entk_distance_prev":          _r(dist_prev),
            "entk_distance_final":         _r(dist_final),
            "mean_future_entk_similarity": _r(mean_future_sim),
            "entk_within_class_mean":      _r(within_mean),
            "entk_between_class_mean":     _r(between_mean),
            "entk_within_between_ratio":   _r(ratio),
            "n_subset":                    n_actual,
            "per_class_subset_size":       n_per_cls,
            "bias_included":               int(has_bias),
        }
        all_result_rows.append(result_row)

        logger.info(
            f"  Epoch {epoch:4d}:  "
            f"dist_prev={dist_prev:.4f}  "
            f"dist_final={dist_final:.4f}  "
            f"mean_future={mean_future_sim:.4f}  "
            f"within_mean={within_mean:.4f}"
        )

    # ------------------------------------------------------------------
    # 11. Save results
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    output_path = os.path.join(results_dir, "entk_summary.csv")
    save_entk_summary_csv(rows=all_result_rows, output_path=output_path, logger=logger)

    logger.info("=" * 70)
    logger.info("eNTK trajectory complete.")
    logger.info(f"  Epochs processed: {n_computed}")
    logger.info(f"  n_subset: {n_actual}  ({n_per_cls} per class × {n_classes} classes)")
    logger.info(f"  p (feature dim): {p}")
    logger.info(f"  Bias included: {has_bias}")
    logger.info(f"  Output: {output_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
