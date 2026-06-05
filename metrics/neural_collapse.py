"""
metrics/neural_collapse.py — Neural Collapse measurements at every stored epoch.

Neural Collapse (NC) is a geometric phenomenon that occurs during the Terminal
Phase of Training (TPT), after training error first reaches zero.  It is
described in Papyan, Han, Donoho (2020) "Prevalence of Neural Collapse during
the terminal phase of deep learning training."

These metrics are SECONDARY EXPLANATORY VARIABLES for this thesis.  They are
not alternative stabilization criteria.  Their purpose is to reveal whether
class-structured geometry in the feature space accompanies, precedes, or
follows CKA stabilization and probe sufficiency — the two primary measurements.

Four NC metrics are computed from the penultimate-layer training representations
at each stored checkpoint epoch:

  NC1  Within-class variability collapse.
       NC1 = trace(Σ_W) / trace(Σ_B)
       Σ_W = within-class scatter (deviation of each training example from its
              class mean), averaged over all training examples.
       Σ_B = between-class scatter (deviation of each class mean from the
              global mean), averaged over all classes.
       NC1 → 0 as within-class variability collapses to zero.
       NC1 is large at early training when features are random.

  NC2  Class-mean alignment to a simplex ETF (equiangular tight frame).
       An ideal ETF satisfies <ĥ_c, ĥ_c'> = −1/(C−1) for every pair c ≠ c',
       where ĥ_c is the normalized, global-mean-centered class mean.
       NC2 = RMS deviation of actual off-diagonal cosine similarities from
             the ideal value −1/(C−1).
       NC2 → 0 at full collapse.

  NC3  Classifier weight / class mean self-duality.
       NC3 = (1/C) Σ_c cos(ŵ_c, ĥ_c)
       ŵ_c = row c of the final FC weight matrix, unit-normalized.
       ĥ_c = centered class mean, unit-normalized.
       NC3 → 1 at full collapse (weights align with class means).

  NC4  Nearest-class-mean (NCM) versus network classifier disagreement.
       For each training example, compare:
         network prediction: argmax over logits W h + b
         NCM prediction:     argmin_c  ||h − μ_c||²
       NC4 = fraction of training examples where the two predictions differ.
       NC4 → 0 at full collapse.

All metrics are computed entirely on training features.  Test features are
not used — NC describes the network's internal geometry on the data it
was trained on.

Inputs (under config["paths"]["activations"])
  full_train/labels.npy               (N,)       — static; loaded once
  full_train/features_epoch_XXXX.npy  (N, 512)   — loaded per epoch

Inputs (under config["paths"]["checkpoints"])
  checkpoint_epoch_XXXX.pt            — fc.weight and fc.bias extracted per epoch

Output (under config["paths"]["results"])
  neural_collapse.csv   — one row per epoch:
    epoch, nc1, log10_nc1, nc2_etf_deviation,
    nc3_weight_mean_alignment, nc4_ncm_disagreement

Usage
  python metrics/neural_collapse.py \\
      --config configs/resnet18_cifar10_200_fullstudy.yaml

  python metrics/neural_collapse.py --config ... --epochs 0 100 200
"""

import argparse
import csv
import logging
import os
import sys

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Reproducibility, logging, config
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("neural_collapse")
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
# Feature file discovery
# ---------------------------------------------------------------------------

def find_feature_epochs(full_train_dir: str) -> list:
    """
    Discover available checkpoint epochs by listing features_epoch_XXXX.npy
    files in the full_train directory.  Returns a numerically sorted list of
    epoch integers.
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


# ---------------------------------------------------------------------------
# Label loading (done once before the epoch loop)
# ---------------------------------------------------------------------------

def load_train_labels(full_train_dir: str, logger: logging.Logger) -> np.ndarray:
    """
    Load the static training-set label array from full_train/labels.npy.
    Labels are produced by features/extract_features.py and do not change
    across epochs; load them once and reuse.

    Returns:
        labels: (N,) int64 array of class labels
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


# ---------------------------------------------------------------------------
# Per-epoch loading
# ---------------------------------------------------------------------------

def load_train_features(epoch: int, full_train_dir: str) -> np.ndarray:
    """
    Load the (N, 512) training feature array for a single checkpoint epoch.
    Raises FileNotFoundError with a clear message if the file is missing.
    """
    feature_path = os.path.join(full_train_dir, f"features_epoch_{epoch:04d}.npy")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(
            f"Training features not found for epoch {epoch}: {feature_path}\n"
            "Run:  python features/extract_features.py --sets full_train"
        )
    features = np.load(feature_path)
    return features


def load_classifier_weights(
    epoch: int,
    checkpoint_dir: str,
    logger: logging.Logger,
) -> tuple:
    """
    Load the final FC layer weight matrix and bias vector from the checkpoint.

    The checkpoint produced by train.py stores the full model_state_dict.
    For ResNet-18 on CIFAR-10, the final FC layer keys are:
      'fc.weight'  shape (n_classes, n_features) = (10, 512)
      'fc.bias'    shape (n_classes,)             = (10,)

    Returns:
        fc_weight: (n_classes, n_features) float64 array
        fc_bias:   (n_classes,)            float64 array
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found for epoch {epoch}: {checkpoint_path}\n"
            "Run train.py first to produce checkpoints."
        )

    logger.info(f"  Loading checkpoint: {checkpoint_path}")

    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_state_dict = checkpoint_data["model_state_dict"]

    fc_weight_tensor = model_state_dict["fc.weight"]
    fc_bias_tensor = model_state_dict["fc.bias"]

    fc_weight = fc_weight_tensor.numpy().astype(np.float64)
    fc_bias = fc_bias_tensor.numpy().astype(np.float64)

    logger.info(
        f"  Classifier weights: fc.weight shape={fc_weight.shape}"
        f"  fc.bias shape={fc_bias.shape}"
    )
    return fc_weight, fc_bias


# ---------------------------------------------------------------------------
# Shared geometry: class means and centered class means
# ---------------------------------------------------------------------------

def compute_class_means(
    features: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    logger: logging.Logger,
) -> tuple:
    """
    Compute the per-class mean and the global mean of the training features.

    Returns:
        class_means:  (n_classes, n_features) — mean feature vector per class
        global_mean:  (n_features,)           — mean over all training examples
        class_counts: (n_classes,) int        — number of training examples per class
    """
    n_features = features.shape[1]

    class_means = np.zeros((n_classes, n_features), dtype=np.float64)
    class_counts = np.zeros(n_classes, dtype=np.int64)

    for class_label in range(n_classes):
        class_mask = (labels == class_label)
        class_count = int(class_mask.sum())
        class_counts[class_label] = class_count

        if class_count == 0:
            logger.warning(f"  Class {class_label} has zero training examples — NC metrics may be unreliable")
            class_means[class_label] = np.zeros(n_features)
        else:
            class_features = features[class_mask].astype(np.float64)
            class_means[class_label] = class_features.mean(axis=0)

    global_mean = features.astype(np.float64).mean(axis=0)   # (n_features,)

    logger.info(
        f"  Class means computed: shape={class_means.shape}"
        f"  global_mean_norm={np.linalg.norm(global_mean):.4f}"
    )
    logger.info(f"  Class counts: {class_counts.tolist()}")
    return class_means, global_mean, class_counts


def compute_centered_normalized_class_means(
    class_means: np.ndarray,
    global_mean: np.ndarray,
    n_classes: int,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Center each class mean by subtracting the global mean, then unit-normalize.

    Returns:
        normalized_centered_class_means: (n_classes, n_features)
            Each row has unit L2 norm (or is the zero vector if the centered
            mean is near-zero, which is logged as a warning).
    """
    epsilon_norm = 1e-10
    n_features = class_means.shape[1]

    centered_class_means = class_means - global_mean       # (n_classes, n_features)
    normalized_centered_class_means = np.zeros_like(centered_class_means)

    for class_label in range(n_classes):
        centered_mean = centered_class_means[class_label]
        norm = float(np.linalg.norm(centered_mean))

        if norm < epsilon_norm:
            logger.warning(
                f"  Class {class_label}: centered class mean has near-zero norm ({norm:.2e}). "
                "NC2 and NC3 for this class are ill-defined. "
                "At epoch 0 this is expected if random features are approximately isotropic."
            )
            normalized_centered_class_means[class_label] = np.zeros(n_features)
        else:
            normalized_centered_class_means[class_label] = centered_mean / norm

    return normalized_centered_class_means


# ---------------------------------------------------------------------------
# NC1: Within-class variability collapse
# ---------------------------------------------------------------------------

def compute_nc1(
    features: np.ndarray,
    labels: np.ndarray,
    class_means: np.ndarray,
    global_mean: np.ndarray,
    n_classes: int,
    logger: logging.Logger,
) -> tuple:
    """
    Compute NC1 = trace(Σ_W) / trace(Σ_B).

    Σ_W (within-class scatter):
      Σ_W = (1/N) * Σ_c Σ_{x in class c} (x − μ_c)(x − μ_c)^T
      trace(Σ_W) = (1/N) * Σ_c Σ_{x in class c} ||x − μ_c||²

    Σ_B (between-class scatter):
      Σ_B = (1/C) * Σ_c (μ_c − μ_G)(μ_c − μ_G)^T
      trace(Σ_B) = (1/C) * Σ_c ||μ_c − μ_G||²

    NC1 → 0 as within-class variability collapses.
    NC1 is large (>> 1) at early training when features are approximately random.

    Returns:
        nc1:       trace(Σ_W) / trace(Σ_B)   (positive float)
        log10_nc1: log10(nc1)                 (can be negative at deep collapse)
    """
    epsilon = 1e-10
    features_f64 = features.astype(np.float64)
    n_total = features_f64.shape[0]

    # ── Within-class scatter (trace only) ────────────────────────────────
    within_class_sum_of_squared_norms = 0.0
    for class_label in range(n_classes):
        class_mask = (labels == class_label)
        class_features = features_f64[class_mask]          # (n_c, n_features)
        class_mean = class_means[class_label]              # (n_features,)

        deviations_from_class_mean = class_features - class_mean   # (n_c, n_features)
        squared_norms = np.sum(deviations_from_class_mean ** 2)    # scalar
        within_class_sum_of_squared_norms += float(squared_norms)

    trace_Sw = within_class_sum_of_squared_norms / n_total

    # ── Between-class scatter (trace only) ───────────────────────────────
    between_class_sum_of_squared_norms = 0.0
    for class_label in range(n_classes):
        centered_class_mean = class_means[class_label] - global_mean   # (n_features,)
        squared_norm = float(np.dot(centered_class_mean, centered_class_mean))
        between_class_sum_of_squared_norms += squared_norm

    trace_Sb = between_class_sum_of_squared_norms / n_classes

    logger.info(f"  NC1: trace(Sw)={trace_Sw:.6f}  trace(Sb)={trace_Sb:.6f}")

    if trace_Sb < epsilon:
        logger.warning(
            f"  NC1: trace(Sb) is near-zero ({trace_Sb:.2e}). "
            "All class means may coincide with the global mean. "
            "NC1 is unreliable; returning large sentinel value."
        )

    nc1 = trace_Sw / (trace_Sb + epsilon)
    log10_nc1 = float(np.log10(nc1 + epsilon))

    logger.info(f"  NC1={nc1:.6f}  log10(NC1)={log10_nc1:.4f}")
    return float(nc1), log10_nc1


# ---------------------------------------------------------------------------
# NC2: ETF deviation of normalized centered class means
# ---------------------------------------------------------------------------

def compute_nc2(
    normalized_centered_class_means: np.ndarray,
    n_classes: int,
    logger: logging.Logger,
) -> float:
    """
    Compute NC2 = RMS deviation of off-diagonal cosine similarities from −1/(C−1).

    For a perfect simplex ETF with C vectors:
      <ĥ_c, ĥ_c'> = −1/(C−1)  for all c ≠ c'

    NC2 measures how far the actual class-mean geometry is from this ideal.
    NC2 → 0 at full Neural Collapse.

    Steps:
      1. Compute the gram matrix G = Ĥ Ĥ^T  (G[c,c'] = cosine similarity)
      2. For each off-diagonal pair (c ≠ c'):
             deviation = G[c, c'] − (−1/(C−1))
      3. NC2 = sqrt(mean of squared deviations over all off-diagonal pairs)

    Returns:
        nc2_etf_deviation: float ≥ 0
    """
    # Step 1: Gram matrix of normalized centered class means
    gram_matrix = normalized_centered_class_means @ normalized_centered_class_means.T
    # gram_matrix shape: (n_classes, n_classes)
    # gram_matrix[c, c'] = cosine similarity between ĥ_c and ĥ_c'

    # Step 2: Ideal off-diagonal cosine similarity for a C-simplex ETF
    ideal_off_diagonal_value = -1.0 / (n_classes - 1)

    # Step 3: Accumulate squared deviations over all off-diagonal pairs
    total_squared_deviation = 0.0
    n_off_diagonal_pairs = 0

    for c in range(n_classes):
        for c_prime in range(n_classes):
            if c == c_prime:
                continue  # skip diagonal (self-similarity = 1 by construction)

            actual_cosine_similarity = float(gram_matrix[c, c_prime])
            deviation_from_ideal = actual_cosine_similarity - ideal_off_diagonal_value
            squared_deviation = deviation_from_ideal ** 2

            total_squared_deviation += squared_deviation
            n_off_diagonal_pairs += 1

    mean_squared_deviation = total_squared_deviation / n_off_diagonal_pairs
    nc2_etf_deviation = float(np.sqrt(mean_squared_deviation))

    logger.info(
        f"  NC2: ideal_off_diag={ideal_off_diagonal_value:.4f}"
        f"  n_pairs={n_off_diagonal_pairs}"
        f"  nc2_etf_deviation={nc2_etf_deviation:.6f}"
    )
    return nc2_etf_deviation


# ---------------------------------------------------------------------------
# NC3: Classifier weight / class mean alignment
# ---------------------------------------------------------------------------

def compute_nc3(
    fc_weight: np.ndarray,
    normalized_centered_class_means: np.ndarray,
    n_classes: int,
    logger: logging.Logger,
) -> float:
    """
    Compute NC3 = (1/C) Σ_c cos(ŵ_c, ĥ_c).

    ŵ_c = row c of fc.weight, unit-normalized.
    ĥ_c = normalized centered class mean (already computed in NC2).

    NC3 → 1 at full Neural Collapse (weights and class means become parallel).
    NC3 is near 0 at early training when weights and features are both random.

    Returns:
        nc3_weight_mean_alignment: float in [−1, 1]
    """
    epsilon_norm = 1e-10

    # Step 1: Normalize each classifier weight row to unit length
    normalized_weight_rows = np.zeros_like(fc_weight, dtype=np.float64)
    for class_label in range(n_classes):
        weight_row = fc_weight[class_label].astype(np.float64)
        weight_norm = float(np.linalg.norm(weight_row))

        if weight_norm < epsilon_norm:
            logger.warning(
                f"  NC3: Classifier weight row {class_label} has near-zero norm ({weight_norm:.2e}). "
                "Alignment for this class is ill-defined."
            )
            normalized_weight_rows[class_label] = np.zeros(fc_weight.shape[1])
        else:
            normalized_weight_rows[class_label] = weight_row / weight_norm

    # Step 2: Compute cosine similarity for each class
    cosine_similarities_per_class = np.zeros(n_classes, dtype=np.float64)
    for class_label in range(n_classes):
        normalized_weight = normalized_weight_rows[class_label]     # (n_features,)
        normalized_mean   = normalized_centered_class_means[class_label]  # (n_features,)

        cosine_similarity = float(np.dot(normalized_weight, normalized_mean))
        cosine_similarities_per_class[class_label] = cosine_similarity

    # Step 3: NC3 is the mean cosine similarity across all classes
    nc3_weight_mean_alignment = float(cosine_similarities_per_class.mean())

    logger.info(
        f"  NC3: per-class cosines={[round(float(v),4) for v in cosine_similarities_per_class]}"
        f"  nc3={nc3_weight_mean_alignment:.6f}"
    )
    return nc3_weight_mean_alignment


# ---------------------------------------------------------------------------
# NC4: Nearest-class-mean versus network classifier disagreement
# ---------------------------------------------------------------------------

def compute_nc4(
    features: np.ndarray,
    labels: np.ndarray,
    class_means: np.ndarray,
    fc_weight: np.ndarray,
    fc_bias: np.ndarray,
    n_classes: int,
    logger: logging.Logger,
) -> float:
    """
    Compute NC4 = fraction of training examples where network prediction
    disagrees with the nearest-class-mean (NCM) prediction.

    Network prediction:  ŷ_net = argmax_c  (W[c] · h + b[c])
    NCM prediction:      ŷ_ncm = argmin_c  ||h − μ_c||²

    For the distance computation, we use the identity:
      ||h − μ_c||² = ||h||² − 2(h · μ_c) + ||μ_c||²
    The ||h||² term is the same for all classes so it does not affect argmin.
    Therefore: ŷ_ncm = argmin_c (−2(h · μ_c) + ||μ_c||²)

    NC4 → 0 when the network classifier and NCM agree on all training examples
    (full Neural Collapse).

    Returns:
        nc4_ncm_disagreement: float in [0, 1]
    """
    features_f64 = features.astype(np.float64)
    fc_weight_f64 = fc_weight.astype(np.float64)
    fc_bias_f64 = fc_bias.astype(np.float64)
    class_means_f64 = class_means.astype(np.float64)

    n_total = features_f64.shape[0]

    # ── Network predictions (argmax of logits) ────────────────────────────
    # logits shape: (n_total, n_classes)
    # logits[i, c] = W[c] · h[i] + b[c]
    logits = features_f64 @ fc_weight_f64.T   # (n_total, n_classes)
    logits = logits + fc_bias_f64             # broadcast bias: (n_total, n_classes)
    network_predictions = np.argmax(logits, axis=1)   # (n_total,)

    # ── NCM predictions (argmin of squared Euclidean distance) ───────────
    # Dot products h_i · μ_c for all i and c simultaneously
    dot_products = features_f64 @ class_means_f64.T   # (n_total, n_classes)

    # Squared norms of class means: ||μ_c||² for each c
    class_mean_squared_norms = np.sum(class_means_f64 ** 2, axis=1)   # (n_classes,)

    # Effective distance proxy (drop the shared ||h||² term):
    #   d²_proxy[i, c] = −2(h_i · μ_c) + ||μ_c||²
    neg_two_dot_products = -2.0 * dot_products           # (n_total, n_classes)
    distance_proxy = neg_two_dot_products + class_mean_squared_norms  # broadcast
    ncm_predictions = np.argmin(distance_proxy, axis=1)  # (n_total,)

    # ── Disagreement fraction ─────────────────────────────────────────────
    n_disagreements = int(np.sum(network_predictions != ncm_predictions))
    nc4_ncm_disagreement = n_disagreements / n_total

    # Sanity check: NCM accuracy on training set
    ncm_train_accuracy = float((ncm_predictions == labels).mean())
    net_train_accuracy = float((network_predictions == labels).mean())

    logger.info(
        f"  NC4: n_disagreements={n_disagreements}/{n_total}"
        f"  nc4={nc4_ncm_disagreement:.6f}"
        f"  net_train_acc={net_train_accuracy:.4f}"
        f"  ncm_train_acc={ncm_train_accuracy:.4f}"
    )
    return float(nc4_ncm_disagreement)


# ---------------------------------------------------------------------------
# CSV saving
# ---------------------------------------------------------------------------

def save_results_csv(
    rows: list,
    output_path: str,
    logger: logging.Logger,
) -> None:
    """Save neural_collapse.csv — one row per checkpoint epoch."""
    fieldnames = [
        "epoch",
        "nc1",
        "log10_nc1",
        "nc2_etf_deviation",
        "nc3_weight_mean_alignment",
        "nc4_ncm_disagreement",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved neural_collapse.csv ({len(rows)} rows)  ->  {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Parse arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Compute Neural Collapse metrics (NC1–NC4) at every stored epoch. "
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
            "Specific epochs to process (e.g. --epochs 0 100 200). "
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
        help="Override the random seed from config.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 2. Load config and resolve paths
    # ------------------------------------------------------------------
    config = load_config(args.config)

    if args.seed is not None:
        config["seed"] = args.seed

    seed = config["seed"]
    activations_dir = args.features_dir or config["paths"]["activations"]
    checkpoint_dir  = args.checkpoints_dir or config["paths"]["checkpoints"]
    results_dir     = args.results_dir or config["paths"]["results"]
    logs_dir        = config["paths"].get("logs", results_dir)

    full_train_dir = os.path.join(activations_dir, "full_train")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Logging
    # ------------------------------------------------------------------
    log_path = os.path.join(logs_dir, "neural_collapse.log")
    logger = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("Neural Collapse Metrics — NC1, NC2, NC3, NC4")
    logger.info("=" * 70)
    logger.info(f"Config:          {args.config}")
    logger.info(f"Activations dir: {activations_dir}")
    logger.info(f"Checkpoint dir:  {checkpoint_dir}")
    logger.info(f"Results dir:     {results_dir}")
    logger.info(f"Seed:            {seed}")

    # ------------------------------------------------------------------
    # 4. Reproducibility
    # ------------------------------------------------------------------
    set_seed(seed)

    # ------------------------------------------------------------------
    # 5. Discover epochs
    # ------------------------------------------------------------------
    if not os.path.isdir(full_train_dir):
        logger.error(
            f"full_train directory not found: {full_train_dir}\n"
            "Run:  python features/extract_features.py --sets full_train"
        )
        sys.exit(1)

    if args.epochs is not None:
        epochs_to_process = sorted(args.epochs)
        logger.info(f"Epochs (from --epochs): {epochs_to_process}")
    else:
        epochs_to_process = find_feature_epochs(full_train_dir)
        logger.info(
            f"Epochs discovered from full_train: {len(epochs_to_process)} checkpoints "
            f"(range {epochs_to_process[0]}–{epochs_to_process[-1]})"
        )

    if len(epochs_to_process) == 0:
        logger.error("No feature epochs found. Run features/extract_features.py first.")
        sys.exit(1)

    logger.info(f"Epoch list: {epochs_to_process}")

    # ------------------------------------------------------------------
    # 6. Load training labels once (static across epochs)
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("Loading training labels (done once — labels are static)...")
    train_labels = load_train_labels(full_train_dir, logger)

    n_total = len(train_labels)
    unique_classes = np.unique(train_labels)
    n_classes = len(unique_classes)

    logger.info(f"Training examples: {n_total}")
    logger.info(f"Classes: {n_classes}  labels: {unique_classes.tolist()}")

    # ------------------------------------------------------------------
    # 7. Main epoch loop
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info(f"Computing NC metrics for {len(epochs_to_process)} epochs...")
    logger.info("=" * 70)

    all_result_rows = []

    for epoch_idx, epoch in enumerate(epochs_to_process):
        logger.info(f"Epoch {epoch:4d}  [{epoch_idx + 1}/{len(epochs_to_process)}]")

        # ── Load training features for this epoch ─────────────────────
        train_features = load_train_features(epoch, full_train_dir)
        n_samples, n_features = train_features.shape
        logger.info(
            f"  Features: shape={train_features.shape}"
            f"  dtype={train_features.dtype}"
            f"  mean={train_features.mean():.4f}"
            f"  std={train_features.std():.4f}"
        )

        # ── Load classifier weights from checkpoint ────────────────────
        fc_weight, fc_bias = load_classifier_weights(epoch, checkpoint_dir, logger)

        # ── Compute class means and global mean ───────────────────────
        class_means, global_mean, class_counts = compute_class_means(
            features=train_features,
            labels=train_labels,
            n_classes=n_classes,
            logger=logger,
        )

        # ── Compute normalized centered class means (used in NC2, NC3) ─
        normalized_centered_class_means = compute_centered_normalized_class_means(
            class_means=class_means,
            global_mean=global_mean,
            n_classes=n_classes,
            logger=logger,
        )

        # ── NC1: within-class variability collapse ────────────────────
        nc1, log10_nc1 = compute_nc1(
            features=train_features,
            labels=train_labels,
            class_means=class_means,
            global_mean=global_mean,
            n_classes=n_classes,
            logger=logger,
        )

        # ── NC2: ETF deviation of class means ─────────────────────────
        nc2_etf_deviation = compute_nc2(
            normalized_centered_class_means=normalized_centered_class_means,
            n_classes=n_classes,
            logger=logger,
        )

        # ── NC3: classifier weight / class mean alignment ─────────────
        nc3_weight_mean_alignment = compute_nc3(
            fc_weight=fc_weight,
            normalized_centered_class_means=normalized_centered_class_means,
            n_classes=n_classes,
            logger=logger,
        )

        # ── NC4: NCM vs network disagreement on training set ──────────
        nc4_ncm_disagreement = compute_nc4(
            features=train_features,
            labels=train_labels,
            class_means=class_means,
            fc_weight=fc_weight,
            fc_bias=fc_bias,
            n_classes=n_classes,
            logger=logger,
        )

        result_row = {
            "epoch": epoch,
            "nc1": round(nc1, 8),
            "log10_nc1": round(log10_nc1, 6),
            "nc2_etf_deviation": round(nc2_etf_deviation, 8),
            "nc3_weight_mean_alignment": round(nc3_weight_mean_alignment, 8),
            "nc4_ncm_disagreement": round(nc4_ncm_disagreement, 8),
        }
        all_result_rows.append(result_row)

        logger.info(
            f"  SUMMARY  nc1={nc1:.4f}  log10_nc1={log10_nc1:.3f}"
            f"  nc2={nc2_etf_deviation:.4f}"
            f"  nc3={nc3_weight_mean_alignment:.4f}"
            f"  nc4={nc4_ncm_disagreement:.4f}"
        )

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------
    output_path = os.path.join(results_dir, "neural_collapse.csv")
    save_results_csv(rows=all_result_rows, output_path=output_path, logger=logger)

    logger.info("=" * 70)
    logger.info("Neural Collapse metrics complete.")
    logger.info(f"Epochs processed: {len(all_result_rows)}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
