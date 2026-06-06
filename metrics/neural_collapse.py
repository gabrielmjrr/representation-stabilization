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

Feature loading: this script reuses penultimate representations extracted by
features/extract_features.py (stored as full_train/features_epoch_XXXX.npy)
and classifier weights from checkpoints produced by train.py.  No additional
feature extraction is performed here.

Four NC metrics are computed from the penultimate-layer training representations
at each stored checkpoint epoch, following Papyan, Han, Donoho (2020):

  NC1  Variability collapse via full p×p scatter matrices.
       Σ_W = (1/N) Σ_c Σ_{i:y_i=c} (h_i − μ_c)(h_i − μ_c)^T      [p×p]
       Σ_B = (1/C) Σ_c (μ_c − μ_G)(μ_c − μ_G)^T                   [p×p]
       nc1_papyan = Tr(Σ_W Σ_B^†) / C
       Σ_B^† is the Moore–Penrose pseudoinverse; Σ_B has rank ≤ C−1 < p
       so a plain inverse does not exist.
       This is NOT the trace-ratio Tr(Σ_W)/Tr(Σ_B).

  NC2  Simplex ETF deviation of normalized centered class means.
       ĥ_c = (μ_c − μ_G) / max(‖μ_c − μ_G‖, ε)
       G = Ĥ Ĥ^T   (C×C Gram matrix of the ĥ vectors)
       G_target = C/(C−1)·I_C − 1/(C−1)·11^T   (ideal ETF Gram)
       nc2_etf_deviation = ‖G − G_target‖_F / ‖G_target‖_F
       Normalisation by ‖G_target‖_F makes the measure scale-invariant.

  NC3  Frobenius self-duality between classifier weights and class means.
       M_dot ∈ R^{p×C}  — centered class means stacked as columns.
       W^T   ∈ R^{p×C}  — transpose of FC weight matrix.
       W_T_norm = W^T / max(‖W‖_F, ε)
       M_norm   = M_dot / max(‖M_dot‖_F, ε)
       nc3_self_duality_frobenius = ‖W_T_norm − M_norm‖_F
       This is NOT the per-class cosine mean; it measures full-matrix alignment.

  NC4  Nearest-class-center (NCC) vs network classifier disagreement.
       pred_net[i]  = argmax_c (W h_i + b)_c
       pred_ncc[i]  = argmin_c ‖h_i − μ_c‖²
                    = argmin_c (−2 h_i·μ_c + ‖μ_c‖²)
       nc4_ncc_disagreement = mean(pred_net ≠ pred_ncc)

Backward-compatibility alias columns are written alongside the canonical
columns so that analysis/build_master_table.py and analysis/plot_main_trajectory.py
require no changes (they read nc1, log10_nc1, nc2_etf_deviation,
nc3_weight_mean_alignment, nc4_ncm_disagreement — present as aliases in output).

Pseudoinverse backend: scipy.linalg.pinv is preferred; falls back to
numpy.linalg.pinv if scipy is not installed.

Inputs (under config["paths"]["activations"])
  full_train/labels.npy               (N,)       — static; loaded once
  full_train/features_epoch_XXXX.npy  (N, 512)   — loaded per epoch

Inputs (under config["paths"]["checkpoints"])
  checkpoint_epoch_XXXX.pt            — fc.weight and fc.bias extracted per epoch

Output (under config["paths"]["results"])
  neural_collapse.csv   — one row per epoch, columns:
    epoch
    nc1_papyan, log10_nc1_papyan        (canonical)
    nc2_etf_deviation                   (canonical)
    nc3_self_duality_frobenius          (canonical)
    nc4_ncc_disagreement                (canonical)
    nc1, log10_nc1                      (backward-compat aliases)
    nc3_weight_mean_alignment           (backward-compat alias)
    nc4_ncm_disagreement                (backward-compat alias)

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

try:
    from scipy.linalg import pinv as matrix_pinv
    _PINV_BACKEND = "scipy"
except ImportError:
    from numpy.linalg import pinv as matrix_pinv
    _PINV_BACKEND = "numpy"


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
    Used for NC2 (Gram matrix computation).

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
                "NC2 is ill-defined for this class. "
                "At epoch 0 this is expected if random features are approximately isotropic."
            )
            normalized_centered_class_means[class_label] = np.zeros(n_features)
        else:
            normalized_centered_class_means[class_label] = centered_mean / norm

    return normalized_centered_class_means


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_nc_inputs(
    features: np.ndarray,
    labels: np.ndarray,
    fc_weight: np.ndarray,
    fc_bias: np.ndarray,
    n_classes: int,
    logger: logging.Logger,
) -> None:
    """
    Validate shapes and class coverage before any NC computation.
    Raises ValueError on hard shape errors.
    Logs a warning for soft issues (near-zero norms are handled per-metric).
    """
    # H: must be 2D [N, p]
    if features.ndim != 2:
        raise ValueError(
            f"features (H) must be 2D [N, p], got ndim={features.ndim} shape={features.shape}"
        )
    N, p = features.shape

    # y: must be 1D [N]
    if labels.ndim != 1:
        raise ValueError(
            f"labels (y) must be 1D [N], got ndim={labels.ndim} shape={labels.shape}"
        )
    if len(labels) != N:
        raise ValueError(
            f"labels length {len(labels)} must equal features rows {N}"
        )

    # W: must be 2D [C, p]
    if fc_weight.ndim != 2:
        raise ValueError(
            f"fc_weight (W) must be 2D [C, p], got ndim={fc_weight.ndim} shape={fc_weight.shape}"
        )
    C_from_W, p_from_W = fc_weight.shape
    if p_from_W != p:
        raise ValueError(
            f"fc_weight columns ({p_from_W}) must match features columns ({p}); "
            "W.T and M_dot would not be compatible"
        )

    # b: must be 1D [C]
    if fc_bias.ndim != 1:
        raise ValueError(
            f"fc_bias (b) must be 1D [C], got ndim={fc_bias.ndim} shape={fc_bias.shape}"
        )
    if len(fc_bias) != C_from_W:
        raise ValueError(
            f"fc_bias length {len(fc_bias)} must match fc_weight rows {C_from_W}"
        )

    # Every class 0..C-1 must appear in y
    unique_labels = sorted(np.unique(labels).tolist())
    expected_labels = list(range(n_classes))
    if unique_labels != expected_labels:
        raise ValueError(
            f"Expected classes {expected_labels} in labels, found {unique_labels}. "
            "All classes 0..C-1 must be present in the training set."
        )

    # C from labels and C from W must agree
    C_from_labels = len(unique_labels)
    if C_from_labels != C_from_W:
        raise ValueError(
            f"n_classes from labels ({C_from_labels}) != n_classes from fc_weight ({C_from_W})"
        )

    logger.info(
        f"  Input validation passed: N={N}  p={p}  C={n_classes}"
        f"  fc_weight={fc_weight.shape}  fc_bias={fc_bias.shape}"
        f"  pinv_backend={_PINV_BACKEND}"
    )


# ---------------------------------------------------------------------------
# NC1 — Papyan variability collapse:  Tr(Σ_W Σ_B^†) / C
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
    NC1 = Tr(Σ_W Σ_B^†) / C   (Papyan, Han, Donoho 2020)

    Σ_W (within-class scatter, p×p):
        Build X_centered — the (N, p) matrix where each row h_i is replaced by
        (h_i − μ_{y_i}).  Then:
            Σ_W = X_centered^T X_centered / N

    Σ_B (between-class scatter, p×p):
        Let M_c ∈ R^{C×p} be the matrix of centered class means (μ_c − μ_G).
            Σ_B = M_c^T M_c / C
        Σ_B has rank ≤ C−1 (for CIFAR-10: rank ≤ 9 out of p=512 dimensions),
        so its pseudoinverse is used instead of a plain inverse.

    Tr(Σ_W Σ_B^†) is computed as np.sum(Σ_W * Σ_B^†), which equals the
    Frobenius inner product <Σ_W, Σ_B^†>_F = Tr(Σ_W^T Σ_B^†) = Tr(Σ_W Σ_B^†)
    because both matrices are symmetric.  This avoids materialising the full
    matrix product and is O(p²) instead of O(p³).

    Returns:
        nc1_papyan:       float ≥ 0
        log10_nc1_papyan: log10(max(nc1_papyan, eps)),  eps = 1e-12
    """
    eps = 1e-12
    H = features.astype(np.float64)     # (N, p)
    N = H.shape[0]

    # ── Within-class scatter Σ_W ─────────────────────────────────────────
    # Subtract each example's class mean in a single vectorised pass.
    X_centered = np.empty_like(H)
    for c in range(n_classes):
        mask = (labels == c)
        X_centered[mask] = H[mask] - class_means[c]   # broadcasting (n_c, p)

    Sigma_W = (X_centered.T @ X_centered) / N          # (p, p), symmetric PSD

    # ── Between-class scatter Σ_B ────────────────────────────────────────
    M_c = (class_means - global_mean).astype(np.float64)   # (C, p) centered means
    Sigma_B = (M_c.T @ M_c) / n_classes                    # (p, p), rank ≤ C−1

    # ── Moore–Penrose pseudoinverse of Σ_B ───────────────────────────────
    Sigma_B_pinv = matrix_pinv(Sigma_B)     # (p, p); backend logged at startup

    # ── NC1 = Tr(Σ_W Σ_B^†) / C  via Frobenius inner product ────────────
    # Both Σ_W and Σ_B^† are symmetric, so Tr(A B) = <A, B>_F = sum(A * B).
    nc1_papyan = float(np.sum(Sigma_W * Sigma_B_pinv)) / n_classes

    log10_nc1_papyan = float(np.log10(max(nc1_papyan, eps)))

    logger.info(
        f"  NC1: Tr(Sw)={float(np.trace(Sigma_W)):.6f}"
        f"  Tr(Sb)={float(np.trace(Sigma_B)):.6f}"
        f"  nc1_papyan={nc1_papyan:.6f}"
        f"  log10={log10_nc1_papyan:.4f}"
    )
    return float(nc1_papyan), log10_nc1_papyan


# ---------------------------------------------------------------------------
# NC2 — Simplex ETF deviation:  ‖G − G_target‖_F / ‖G_target‖_F
# ---------------------------------------------------------------------------

def compute_nc2(
    normalized_centered_class_means: np.ndarray,
    n_classes: int,
    logger: logging.Logger,
) -> float:
    """
    NC2 = ‖G − G_target‖_F / ‖G_target‖_F

    G ∈ R^{C×C} is the Gram matrix of the unit-normalized centered class means:
        G = Ĥ Ĥ^T,   G[c, c'] = ĥ_c · ĥ_{c'}

    G_target is the Gram matrix of an ideal C-class simplex ETF:
        G_target = C/(C−1) · I_C − 1/(C−1) · 11^T
    Its diagonal entries are 1 and its off-diagonal entries are −1/(C−1).

    Normalising by ‖G_target‖_F makes NC2 dimensionless and comparable across
    different values of C.

    NC2 → 0 at full Neural Collapse.
    """
    C = n_classes

    # Gram matrix of normalized centered class means (C×C)
    G = normalized_centered_class_means @ normalized_centered_class_means.T

    # Ideal simplex ETF Gram matrix
    G_target = (C / (C - 1)) * np.eye(C) - (1.0 / (C - 1)) * np.ones((C, C))
    # Diagonal: C/(C-1) - 1/(C-1) = 1.  Off-diagonal: -1/(C-1).

    norm_G_target = float(np.linalg.norm(G_target, "fro"))
    if norm_G_target < 1e-12:
        logger.warning("  NC2: ‖G_target‖_F is near-zero; returning 0.")
        return 0.0

    nc2_etf_deviation = float(np.linalg.norm(G - G_target, "fro")) / norm_G_target

    logger.info(
        f"  NC2: ‖G‖_F={float(np.linalg.norm(G, 'fro')):.4f}"
        f"  ‖G_target‖_F={norm_G_target:.4f}"
        f"  nc2_etf_deviation={nc2_etf_deviation:.6f}"
    )
    return nc2_etf_deviation


# ---------------------------------------------------------------------------
# NC3 — Frobenius self-duality:  ‖W_T_norm − M_norm‖_F
# ---------------------------------------------------------------------------

def compute_nc3(
    fc_weight: np.ndarray,
    centered_class_means: np.ndarray,
    n_classes: int,
    logger: logging.Logger,
) -> float:
    """
    NC3 = ‖W_T_norm − M_norm‖_F

    W^T ∈ R^{p×C}    transpose of the FC weight matrix W ∈ R^{C×p}.
    M_dot ∈ R^{p×C}  centered class means stacked as columns:
                      M_dot[:, c] = μ_c − μ_G   for c = 0..C−1.

    Both are normalised globally by their Frobenius norms before comparison:
        W_T_norm = W^T / max(‖W‖_F, ε)
        M_norm   = M_dot / max(‖M_dot‖_F, ε)

    NC3 → 0 at full Neural Collapse (the weight matrix and class-mean matrix
    become proportional, so W_T_norm = M_norm).

    Global Frobenius normalisation captures the full geometric relationship
    between W and the class means, not just per-class angular alignment.
    """
    eps = 1e-12
    W = fc_weight.astype(np.float64)              # (C, p)
    W_T = W.T                                      # (p, C)

    # M_dot = centered class means as columns: shape (p, C)
    M_dot = centered_class_means.astype(np.float64).T   # (p, C)

    norm_W = float(np.linalg.norm(W, "fro"))
    norm_M = float(np.linalg.norm(M_dot, "fro"))

    if norm_W < eps:
        logger.warning(f"  NC3: ‖W‖_F near-zero ({norm_W:.2e}); W_T_norm set to zeros.")
    if norm_M < eps:
        logger.warning(f"  NC3: ‖M_dot‖_F near-zero ({norm_M:.2e}); M_norm set to zeros.")

    W_T_norm = W_T / max(norm_W, eps)
    M_norm   = M_dot / max(norm_M, eps)

    nc3_self_duality_frobenius = float(np.linalg.norm(W_T_norm - M_norm, "fro"))

    logger.info(
        f"  NC3: ‖W‖_F={norm_W:.4f}"
        f"  ‖M_dot‖_F={norm_M:.4f}"
        f"  nc3_self_duality_frobenius={nc3_self_duality_frobenius:.6f}"
    )
    return nc3_self_duality_frobenius


# ---------------------------------------------------------------------------
# NC4 — Nearest-class-center vs network classifier disagreement
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
    NC4 = mean(pred_net ≠ pred_ncc)  over all N training examples.

    Network prediction:
        logits[i, c] = W[c] · h_i + b[c]
        pred_net[i]  = argmax_c logits[i]

    Nearest-class-center (NCC) prediction uses the identity
        ‖h_i − μ_c‖² = ‖h_i‖² − 2(h_i·μ_c) + ‖μ_c‖²
    The ‖h_i‖² term is identical for all c and does not affect argmin:
        pred_ncc[i]  = argmin_c (−2 h_i·μ_c + ‖μ_c‖²)

    NC4 → 0 when the network classifier and NCC agree on every training example.
    """
    H   = features.astype(np.float64)     # (N, p)
    W   = fc_weight.astype(np.float64)    # (C, p)
    b   = fc_bias.astype(np.float64)      # (C,)
    Mu  = class_means.astype(np.float64)  # (C, p)  uncentered class means

    N = H.shape[0]

    # Network predictions: argmax of logits W h + b
    logits   = H @ W.T + b                        # (N, C)
    pred_net = np.argmax(logits, axis=1)           # (N,)

    # NCC predictions: argmin of squared-distance proxy
    dot_products        = H @ Mu.T                 # (N, C)  h_i · μ_c
    class_mean_sq_norms = np.sum(Mu ** 2, axis=1)  # (C,)    ‖μ_c‖²
    dist_proxy          = -2.0 * dot_products + class_mean_sq_norms   # (N, C)
    pred_ncc            = np.argmin(dist_proxy, axis=1)  # (N,)

    n_disagreements      = int(np.sum(pred_net != pred_ncc))
    nc4_ncc_disagreement = n_disagreements / N

    ncc_train_acc = float((pred_ncc == labels).mean())
    net_train_acc = float((pred_net == labels).mean())

    logger.info(
        f"  NC4: n_disagreements={n_disagreements}/{N}"
        f"  nc4={nc4_ncc_disagreement:.6f}"
        f"  net_train_acc={net_train_acc:.4f}"
        f"  ncc_train_acc={ncc_train_acc:.4f}"
    )
    return float(nc4_ncc_disagreement)


# ---------------------------------------------------------------------------
# CSV saving
# ---------------------------------------------------------------------------

def save_results_csv(
    rows: list,
    output_path: str,
    logger: logging.Logger,
) -> None:
    """
    Save neural_collapse.csv — one row per checkpoint epoch.

    Primary canonical columns (new names per Papyan et al. 2020):
      epoch, nc1_papyan, log10_nc1_papyan, nc2_etf_deviation,
      nc3_self_duality_frobenius, nc4_ncc_disagreement

    Backward-compatibility alias columns (old names, same values):
      nc1             = nc1_papyan
      log10_nc1       = log10_nc1_papyan
      nc2_etf_deviation  (same name — no alias needed)
      nc3_weight_mean_alignment = nc3_self_duality_frobenius
      nc4_ncm_disagreement      = nc4_ncc_disagreement

    Downstream scripts read the alias columns and require no changes.
    """
    fieldnames = [
        "epoch",
        # Canonical columns
        "nc1_papyan",
        "log10_nc1_papyan",
        "nc2_etf_deviation",
        "nc3_self_duality_frobenius",
        "nc4_ncc_disagreement",
        # Backward-compatibility aliases
        "nc1",
        "log10_nc1",
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
    logger.info("Neural Collapse Metrics — NC1, NC2, NC3, NC4 (Papyan et al. 2020)")
    logger.info("=" * 70)
    logger.info(f"Config:          {args.config}")
    logger.info(f"Activations dir: {activations_dir}")
    logger.info(f"Checkpoint dir:  {checkpoint_dir}")
    logger.info(f"Results dir:     {results_dir}")
    logger.info(f"Seed:            {seed}")
    logger.info(f"Pseudoinverse:   {_PINV_BACKEND}.linalg.pinv")

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

        # ── Validate inputs (shapes, class coverage, backend) ─────────
        try:
            validate_nc_inputs(
                features=train_features,
                labels=train_labels,
                fc_weight=fc_weight,
                fc_bias=fc_bias,
                n_classes=n_classes,
                logger=logger,
            )
        except ValueError as exc:
            logger.error(f"  Input validation FAILED for epoch {epoch}: {exc}")
            logger.error("  Skipping NC metrics for this epoch.")
            continue

        # ── Compute class means and global mean ───────────────────────
        class_means, global_mean, class_counts = compute_class_means(
            features=train_features,
            labels=train_labels,
            n_classes=n_classes,
            logger=logger,
        )

        # ── Raw centered class means (C×p) — used for NC1 and NC3 ─────
        # M_c[c] = μ_c − μ_G;  M_dot = M_c.T has shape (p, C)
        centered_class_means = (class_means - global_mean).astype(np.float64)

        # ── Per-row unit-normalized centered means — used for NC2 ──────
        normalized_centered_class_means = compute_centered_normalized_class_means(
            class_means=class_means,
            global_mean=global_mean,
            n_classes=n_classes,
            logger=logger,
        )

        # ── NC1: Tr(Σ_W Σ_B^†) / C ───────────────────────────────────
        nc1_papyan, log10_nc1_papyan = compute_nc1(
            features=train_features,
            labels=train_labels,
            class_means=class_means,
            global_mean=global_mean,
            n_classes=n_classes,
            logger=logger,
        )

        # ── NC2: ‖G − G_target‖_F / ‖G_target‖_F ────────────────────
        nc2_etf_deviation = compute_nc2(
            normalized_centered_class_means=normalized_centered_class_means,
            n_classes=n_classes,
            logger=logger,
        )

        # ── NC3: ‖W_T_norm − M_norm‖_F ───────────────────────────────
        nc3_self_duality_frobenius = compute_nc3(
            fc_weight=fc_weight,
            centered_class_means=centered_class_means,
            n_classes=n_classes,
            logger=logger,
        )

        # ── NC4: mean(pred_net ≠ pred_ncc) ────────────────────────────
        nc4_ncc_disagreement = compute_nc4(
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
            # Canonical columns
            "nc1_papyan":                 round(nc1_papyan, 8),
            "log10_nc1_papyan":           round(log10_nc1_papyan, 6),
            "nc2_etf_deviation":          round(nc2_etf_deviation, 8),
            "nc3_self_duality_frobenius": round(nc3_self_duality_frobenius, 8),
            "nc4_ncc_disagreement":       round(nc4_ncc_disagreement, 8),
            # Backward-compatibility aliases (read by build_master_table / plot scripts)
            "nc1":                        round(nc1_papyan, 8),
            "log10_nc1":                  round(log10_nc1_papyan, 6),
            "nc3_weight_mean_alignment":  round(nc3_self_duality_frobenius, 8),
            "nc4_ncm_disagreement":       round(nc4_ncc_disagreement, 8),
        }
        all_result_rows.append(result_row)

        logger.info(
            f"  SUMMARY  nc1_papyan={nc1_papyan:.4f}  log10={log10_nc1_papyan:.3f}"
            f"  nc2={nc2_etf_deviation:.4f}"
            f"  nc3={nc3_self_duality_frobenius:.4f}"
            f"  nc4={nc4_ncc_disagreement:.4f}"
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
