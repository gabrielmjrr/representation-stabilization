"""
surrogates/run_probes.py — Unified surrogate evaluation framework.

Fits five surrogate classifiers on frozen penultimate-layer features at every
stored checkpoint epoch and writes per-epoch accuracy to CSV.

This is one of the primary thesis measurements.  Probe test accuracy at each
epoch is compared against CKA stabilization metrics to determine whether
representation stabilization implies representation sufficiency.

Surrogate classifiers
  logistic_regression — Logistic Regression (L-BFGS, all 50 k train examples)
  linear_svm          — Linear SVM via LinearSVC on a fixed stratified subset of train
                        (default 10 000; full 50 k is too slow per checkpoint.
                        Subset indices are saved once.)
  rbf_svm             — RBF kernel SVM on a fixed stratified subset of train
                        (default 10 000; full 50 k makes the kernel matrix
                        intractable.  Subset indices are saved once.)
  random_forest       — Random Forest (all 50 k train examples, raw features)
  lightgbm            — LightGBM gradient boosting (all 50 k, raw features)

Preprocessing rules — no train/test leakage
  Logistic Regression, Linear SVM, RBF SVM:
    StandardScaler is fit ONLY on the full 50 k training features.
    The same fitted scaler transforms the training subset (for RBF SVM)
    and the test features.  scaler.fit() is never called on test data.
  Random Forest, LightGBM:
    Raw unscaled features.  Tree-based models are scale-invariant.

Inputs (under config["paths"]["activations"])
  full_train/labels.npy               (50000,) — static across epochs
  full_train/features_epoch_XXXX.npy  (50000, 512) — one file per epoch
  full_test/labels.npy                (10000,) — static across epochs
  full_test/features_epoch_XXXX.npy   (10000, 512) — one file per epoch

Outputs (under config["paths"]["results"])
  probe_results_long.csv  — one row per (epoch × probe)
  probe_results_wide.csv  — one row per epoch, probes as columns (test_acc)
  rbf_svm_train_indices.json    — fixed stratified indices into the train set (RBF SVM)
  linear_svm_train_indices.json — fixed stratified indices into the train set (Linear SVM)

Usage
  python surrogates/run_probes.py \\
    --config configs/resnet18_cifar10_200_fullstudy.yaml

  # Only specific epochs
  python surrogates/run_probes.py --config ... --epochs 0 5 10 50 100 200

  # Only specific probes
  python surrogates/run_probes.py --config ... --probes logistic_regression lightgbm
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Canonical probe names and their corresponding wide-CSV column names
ALL_PROBE_NAMES = [
    "logistic_regression",
    "linear_svm",
    "rbf_svm",
    "random_forest",
    "lightgbm",
]

PROBE_WIDE_COLUMN = {
    "logistic_regression": "logistic_acc",
    "linear_svm":          "linear_svm_acc",
    "rbf_svm":             "rbf_svm_acc",
    "random_forest":       "rf_acc",
    "lightgbm":            "lightgbm_acc",
}


# ---------------------------------------------------------------------------
# Reproducibility, logging, config
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("run_probes")
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
    Discover available epochs by listing features_epoch_XXXX.npy files
    in the full_train feature directory.  Returns a numerically sorted list
    of epoch integers.
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
# Feature and label loading
# ---------------------------------------------------------------------------

def load_labels_once(
    full_train_dir: str,
    full_test_dir: str,
    logger: logging.Logger,
) -> tuple:
    """
    Load train and test labels from their static label files.
    Labels are the same across all epochs; load them once before the epoch loop.

    Returns:
        train_labels: (n_train,) int64 array
        test_labels:  (n_test,)  int64 array
    """
    train_labels_path = os.path.join(full_train_dir, "labels.npy")
    test_labels_path = os.path.join(full_test_dir, "labels.npy")

    if not os.path.exists(train_labels_path):
        raise FileNotFoundError(
            f"Train labels not found: {train_labels_path}\n"
            "Run: python features/extract_features.py --sets full_train"
        )
    if not os.path.exists(test_labels_path):
        raise FileNotFoundError(
            f"Test labels not found: {test_labels_path}\n"
            "Run: python features/extract_features.py --sets full_test"
        )

    train_labels = np.load(train_labels_path).astype(np.int64)
    test_labels = np.load(test_labels_path).astype(np.int64)

    logger.info(f"Train labels: shape={train_labels.shape}  classes={np.unique(train_labels).tolist()}")
    logger.info(f"Test  labels: shape={test_labels.shape}  classes={np.unique(test_labels).tolist()}")
    return train_labels, test_labels


def load_features_for_epoch(
    epoch: int,
    full_train_dir: str,
    full_test_dir: str,
) -> tuple:
    """
    Load train and test feature arrays for a single checkpoint epoch.

    Returns:
        X_train: (n_train, n_features) float32 array — raw, unscaled
        X_test:  (n_test,  n_features) float32 array — raw, unscaled
    """
    train_features_path = os.path.join(full_train_dir, f"features_epoch_{epoch:04d}.npy")
    test_features_path = os.path.join(full_test_dir, f"features_epoch_{epoch:04d}.npy")

    if not os.path.exists(train_features_path):
        raise FileNotFoundError(
            f"Train features not found for epoch {epoch}: {train_features_path}\n"
            "Run: python features/extract_features.py --sets full_train"
        )
    if not os.path.exists(test_features_path):
        raise FileNotFoundError(
            f"Test features not found for epoch {epoch}: {test_features_path}\n"
            "Run: python features/extract_features.py --sets full_test"
        )

    X_train = np.load(train_features_path)
    X_test = np.load(test_features_path)
    return X_train, X_test


# ---------------------------------------------------------------------------
# StandardScaler — no leakage
# ---------------------------------------------------------------------------

def scale_features_no_leakage(
    X_train: np.ndarray,
    X_test: np.ndarray,
    logger: logging.Logger,
) -> tuple:
    """
    Fit StandardScaler on train features only; transform both train and test.

    The scaler is NEVER fitted on test data.  Fitting on test would leak
    test distribution information into the preprocessing, inflating test
    accuracy and invalidating the sufficiency measurement.

    Returns:
        X_train_scaled: (n_train, n_features) float64 — zero mean, unit variance
        X_test_scaled:  (n_test, n_features)  float64 — same transform as train
        scaler:         the fitted StandardScaler (for inspection/reproducibility)
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)    # fit on train, then transform
    X_test_scaled = scaler.transform(X_test)           # transform only — no fit on test

    logger.info(
        f"  Scaler fit on train (n={X_train.shape[0]})  "
        f"mean range [{X_train_scaled.mean(axis=0).min():.3f}, "
        f"{X_train_scaled.mean(axis=0).max():.3f}]  "
        f"(test transformed without re-fitting)"
    )
    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# RBF SVM train subset
# ---------------------------------------------------------------------------

def compute_rbf_subset_indices(
    train_labels: np.ndarray,
    n_subset: int,
    seed: int,
) -> list:
    """
    Compute a fixed stratified subset of training indices for RBF SVM.

    Stratified: n_subset // n_classes examples per class.
    Fixed: the same seed always produces the same indices.

    The subset is computed from train_labels (static across epochs), so it
    can be computed once and reused for every epoch without loading features.

    Returns:
        selected_indices: sorted list of int indices into the train set
    """
    n_classes = len(np.unique(train_labels))
    n_per_class = n_subset // n_classes

    rng = np.random.RandomState(seed)
    selected = []
    for class_label in range(n_classes):
        class_indices = np.where(train_labels == class_label)[0]
        sampled = rng.choice(class_indices, size=n_per_class, replace=False)
        selected.extend(sampled.tolist())

    selected.sort()   # sorted for consistent ordering
    return selected


def save_rbf_subset_indices(
    indices: list,
    results_dir: str,
    logger: logging.Logger,
) -> str:
    """
    Save RBF SVM train subset indices to a JSON file.
    JSON is used so the file is text-based and can be committed to git.
    Skips writing if the file already exists (indices are deterministic).
    Returns the path to the saved file.
    """
    indices_path = os.path.join(results_dir, "rbf_svm_train_indices.json")
    if not os.path.exists(indices_path):
        with open(indices_path, "w") as f:
            json.dump(indices, f)
        logger.info(f"Saved RBF SVM train subset indices ({len(indices)} examples): {indices_path}")
    else:
        logger.info(f"RBF SVM train subset indices already saved: {indices_path}")
    return indices_path


# ---------------------------------------------------------------------------
# Linear SVM train subset
# ---------------------------------------------------------------------------

def compute_linear_svm_subset_indices(
    train_labels: np.ndarray,
    n_subset: int,
    seed: int,
) -> list:
    """
    Compute a fixed stratified subset of training indices for Linear SVM.

    Same stratified sampling logic as compute_rbf_subset_indices.
    Fixed: the same seed always produces the same indices.

    Returns:
        selected_indices: sorted list of int indices into the train set
    """
    n_classes = len(np.unique(train_labels))
    n_per_class = n_subset // n_classes

    rng = np.random.RandomState(seed)
    selected = []
    for class_label in range(n_classes):
        class_indices = np.where(train_labels == class_label)[0]
        sampled = rng.choice(class_indices, size=n_per_class, replace=False)
        selected.extend(sampled.tolist())

    selected.sort()
    return selected


def save_linear_svm_subset_indices(
    indices: list,
    results_dir: str,
    logger: logging.Logger,
) -> str:
    """
    Save Linear SVM train subset indices to a JSON file.
    Skips writing if the file already exists (indices are deterministic).
    Returns the path to the saved file.
    """
    indices_path = os.path.join(results_dir, "linear_svm_train_indices.json")
    if not os.path.exists(indices_path):
        with open(indices_path, "w") as f:
            json.dump(indices, f)
        logger.info(f"Saved Linear SVM train subset indices ({len(indices)} examples): {indices_path}")
    else:
        logger.info(f"Linear SVM train subset indices already saved: {indices_path}")
    return indices_path


# ---------------------------------------------------------------------------
# Individual probe fitters
# ---------------------------------------------------------------------------

def fit_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
    seed: int,
) -> tuple:
    """
    Fit Logistic Regression and return (train_acc, test_acc, fit_seconds).
    Inputs must be SCALED features (StandardScaler already applied).
    """
    lr_config = config["surrogates"]["logistic_regression"]
    C = lr_config["C"]
    max_iter = lr_config["max_iter"]
    solver = lr_config.get("solver", "lbfgs")

    clf = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=seed,
        verbose=0,
    )

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0

    train_predictions = clf.predict(X_train)
    test_predictions = clf.predict(X_test)

    train_acc = float((train_predictions == y_train).mean())
    test_acc = float((test_predictions == y_test).mean())

    return train_acc, test_acc, fit_seconds


def fit_linear_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
    seed: int,
) -> tuple:
    """
    Fit a Linear SVM via LinearSVC and return (train_acc, test_acc, fit_seconds).
    Inputs must be SCALED features. X_train may be a stratified subset of train
    (when n_train_subset is configured) rather than the full 50 k set.

    LinearSVC is used rather than SVC(kernel='linear') because LinearSVC uses a
    liblinear implementation that is much faster for large n_samples.
    dual=False is set because n_samples > n_features; this is more efficient.
    """
    svm_config = config["surrogates"]["linear_svm"]
    C = svm_config["C"]
    max_iter = svm_config["max_iter"]

    clf = LinearSVC(
        C=C,
        max_iter=max_iter,
        dual=False,       # n_samples (50 000) >> n_features (512) — primal form is faster
        random_state=seed,
        verbose=0,
    )

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0

    train_predictions = clf.predict(X_train)
    test_predictions = clf.predict(X_test)

    train_acc = float((train_predictions == y_train).mean())
    test_acc = float((test_predictions == y_test).mean())

    return train_acc, test_acc, fit_seconds


def fit_rbf_svm(
    X_train_subset: np.ndarray,
    y_train_subset: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
    seed: int,
) -> tuple:
    """
    Fit an RBF SVM on the pre-selected train subset and return
    (train_acc, test_acc, fit_seconds).

    X_train_subset must be SCALED (scaler fitted on the FULL train set).
    X_test must be SCALED with the same scaler.
    train_acc is computed on the subset only (the training data this SVM saw).
    test_acc is computed on the full test set.

    n_train reported in the long CSV will be len(X_train_subset), not 50 000.
    """
    rbf_config = config["surrogates"]["rbf_svm"]
    C = rbf_config["C"]
    gamma = rbf_config["gamma"]
    max_iter = rbf_config["max_iter"]

    clf = SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        max_iter=max_iter,
        random_state=seed,
        verbose=False,
    )

    t0 = time.perf_counter()
    clf.fit(X_train_subset, y_train_subset)
    fit_seconds = time.perf_counter() - t0

    train_predictions = clf.predict(X_train_subset)
    test_predictions = clf.predict(X_test)

    train_acc = float((train_predictions == y_train_subset).mean())
    test_acc = float((test_predictions == y_test).mean())

    return train_acc, test_acc, fit_seconds


def fit_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
    seed: int,
) -> tuple:
    """
    Fit a Random Forest on raw (unscaled) features and return
    (train_acc, test_acc, fit_seconds).

    Tree-based models are scale-invariant; scaling would have no effect
    and is omitted to reduce unnecessary computation.
    """
    rf_config = config["surrogates"]["random_forest"]
    n_estimators = rf_config["n_estimators"]
    max_features = rf_config["max_features"]
    n_jobs = rf_config.get("n_jobs", -1)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        n_jobs=n_jobs,
        random_state=seed,
        verbose=0,
    )

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0

    train_predictions = clf.predict(X_train)
    test_predictions = clf.predict(X_test)

    train_acc = float((train_predictions == y_train).mean())
    test_acc = float((test_predictions == y_test).mean())

    return train_acc, test_acc, fit_seconds


def fit_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
    seed: int,
) -> tuple:
    """
    Fit a LightGBM classifier on raw (unscaled) features and return
    (train_acc, test_acc, fit_seconds).

    LightGBM is gradient boosting — also scale-invariant for tree splits.
    verbose=-1 suppresses LightGBM's stdout progress output.
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError(
            "lightgbm is not installed. Install with: pip install lightgbm"
        )

    lgb_config = config["surrogates"]["lightgbm"]
    n_estimators = lgb_config["n_estimators"]
    num_leaves = lgb_config["num_leaves"]
    learning_rate = lgb_config["learning_rate"]
    n_jobs = lgb_config.get("n_jobs", -1)
    max_n_estimators = lgb_config.get("max_n_estimators", None)
    if max_n_estimators is not None:
        n_estimators = min(n_estimators, max_n_estimators)

    clf = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        n_jobs=n_jobs,
        random_state=seed,
        verbose=-1,
        force_col_wise=True,
    )

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0

    train_predictions = clf.predict(X_train)
    test_predictions = clf.predict(X_test)

    train_acc = float((train_predictions == y_train).mean())
    test_acc = float((test_predictions == y_test).mean())

    return train_acc, test_acc, fit_seconds


# ---------------------------------------------------------------------------
# Per-probe dispatcher with error isolation
# ---------------------------------------------------------------------------

def run_single_probe(
    probe_name: str,
    probe_fn,
    fn_args: tuple,
    logger: logging.Logger,
) -> tuple:
    """
    Call probe_fn(*fn_args) inside a try/except.

    On success: return (train_acc, test_acc, fit_seconds, "ok", "")
    On failure: log the error, return (nan, nan, nan, "failed", error_message)

    The calling loop continues regardless of the return value — one failed
    probe does not abort the remaining probes for the same epoch.
    """
    try:
        logger.info(f"    [{probe_name}] Starting fit...")
        train_acc, test_acc, fit_seconds = probe_fn(*fn_args)
        status = "ok"
        error_message = ""
        logger.info(
            f"    [{probe_name}]  train_acc={train_acc:.4f}"
            f"  test_acc={test_acc:.4f}"
            f"  fit={fit_seconds:.1f}s"
        )

    except Exception:
        error_text = traceback.format_exc()
        short_error = error_text.strip().split("\n")[-1][:500]
        logger.error(f"    [{probe_name}]  FAILED: {short_error}")
        logger.debug(error_text)

        train_acc = float("nan")
        test_acc = float("nan")
        fit_seconds = float("nan")
        status = "failed"
        error_message = short_error

    return train_acc, test_acc, fit_seconds, status, error_message


# ---------------------------------------------------------------------------
# CSV saving
# ---------------------------------------------------------------------------

def save_long_csv(
    rows: list,
    output_path: str,
    logger: logging.Logger,
) -> None:
    """
    Save probe_results_long.csv.
    One row per (epoch × probe).  Includes train_acc, test_acc, fit time, status.
    """
    fieldnames = [
        "epoch",
        "probe_name",
        "train_acc",
        "test_acc",
        "n_train",
        "n_test",
        "fit_seconds",
        "status",
        "error_message",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved long-format results ({len(rows)} rows)  ->  {output_path}")


def save_wide_csv(
    rows: list,
    output_path: str,
    logger: logging.Logger,
) -> None:
    """
    Save probe_results_wide.csv.
    One row per epoch; each probe's test_acc is a separate column.
    nan for probes that failed or were not requested.
    """
    fieldnames = ["epoch"] + [PROBE_WIDE_COLUMN[p] for p in ALL_PROBE_NAMES]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved wide-format results ({len(rows)} rows)  ->  {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Parse arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Fit surrogate classifiers on frozen features at every checkpoint epoch "
            "and write probe accuracy results to CSV. "
            "Run features/extract_features.py --sets full_train full_test first."
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
            "Specific epochs to evaluate (e.g. --epochs 0 5 10 100 200). "
            "Default: all epochs found in the full_train feature directory."
        ),
    )
    parser.add_argument(
        "--probes",
        type=str,
        nargs="+",
        choices=ALL_PROBE_NAMES,
        default=ALL_PROBE_NAMES,
        help="Subset of probes to fit. Default: all five.",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default=None,
        help="Override the activations base directory from config.",
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
    results_dir = args.results_dir or config["paths"]["results"]
    logs_dir = config["paths"].get("logs", results_dir)

    full_train_dir = os.path.join(activations_dir, "full_train")
    full_test_dir = os.path.join(activations_dir, "full_test")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Logging
    # ------------------------------------------------------------------
    log_path = os.path.join(logs_dir, "run_probes.log")
    logger = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("Surrogate Evaluation — Representation Sufficiency")
    logger.info("=" * 70)
    logger.info(f"Config:          {args.config}")
    logger.info(f"Activations dir: {activations_dir}")
    logger.info(f"Results dir:     {results_dir}")
    logger.info(f"Seed:            {seed}")
    logger.info(f"Probes:          {args.probes}")

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
            "Run: python features/extract_features.py --sets full_train"
        )
        sys.exit(1)

    discovered_epochs = find_feature_epochs(full_train_dir)

    if args.epochs is not None:
        epochs_to_process = sorted(args.epochs)
        logger.info(f"Epochs (from --epochs arg): {epochs_to_process}")
    else:
        validate_feature_epochs(
            found_epochs=discovered_epochs,
            expected_epochs=config["checkpoint_epochs"],
            logger=logger,
        )
        epochs_to_process = discovered_epochs
        logger.info(
            f"Epochs discovered in full_train: {len(epochs_to_process)} "
            f"(range {epochs_to_process[0]}–{epochs_to_process[-1]})"
        )

    if len(epochs_to_process) == 0:
        logger.error("No feature epochs found. Run features/extract_features.py first.")
        sys.exit(1)

    logger.info(f"Epoch list: {epochs_to_process}")

    # ------------------------------------------------------------------
    # 6. Load labels once (static across epochs)
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("Loading labels (done once — labels are static across epochs)...")
    train_labels, test_labels = load_labels_once(full_train_dir, full_test_dir, logger)

    n_train_full = len(train_labels)
    n_test = len(test_labels)

    # ------------------------------------------------------------------
    # 7. Compute RBF SVM train subset indices (once, from labels)
    # ------------------------------------------------------------------
    rbf_svm_config = config["surrogates"]["rbf_svm"]
    n_rbf_subset = rbf_svm_config.get("n_train_subset", 10000)

    rbf_train_indices = compute_rbf_subset_indices(
        train_labels=train_labels,
        n_subset=n_rbf_subset,
        seed=seed,
    )
    n_rbf_actual = len(rbf_train_indices)
    logger.info(f"RBF SVM train subset: {n_rbf_actual} examples (stratified from {n_train_full})")

    save_rbf_subset_indices(rbf_train_indices, results_dir, logger)

    # Subset labels (static — same examples every epoch)
    y_rbf_train = train_labels[rbf_train_indices]

    # ------------------------------------------------------------------
    # 8. Compute Linear SVM train subset indices (once, from labels)
    # ------------------------------------------------------------------
    linear_svm_config = config["surrogates"]["linear_svm"]
    n_linear_subset = linear_svm_config.get("n_train_subset", None)

    if n_linear_subset is not None:
        linear_train_indices = compute_linear_svm_subset_indices(
            train_labels=train_labels,
            n_subset=n_linear_subset,
            seed=seed,
        )
        n_linear_actual = len(linear_train_indices)
        logger.info(
            f"Linear SVM train subset: {n_linear_actual} examples "
            f"(stratified from {n_train_full})"
        )
        save_linear_svm_subset_indices(linear_train_indices, results_dir, logger)
        y_linear_train = train_labels[linear_train_indices]
    else:
        linear_train_indices = None
        n_linear_actual = n_train_full
        y_linear_train = train_labels
        logger.info("Linear SVM train subset: not configured — using full training set")

    # ------------------------------------------------------------------
    # 9. Main epoch loop
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info(f"Evaluating {len(args.probes)} probes across {len(epochs_to_process)} epochs...")
    logger.info("=" * 70)

    all_long_rows = []
    all_wide_rows = []

    for epoch_idx, epoch in enumerate(epochs_to_process):
        logger.info(
            f"Epoch {epoch:4d}  [{epoch_idx + 1}/{len(epochs_to_process)}]"
        )

        # ── Load features for this epoch ─────────────────────────────
        try:
            X_train_raw, X_test_raw = load_features_for_epoch(
                epoch=epoch,
                full_train_dir=full_train_dir,
                full_test_dir=full_test_dir,
            )
        except FileNotFoundError as e:
            logger.error(f"  Skipping epoch {epoch}: {e}")
            # Write a failed row for every probe at this epoch
            for probe_name in args.probes:
                all_long_rows.append({
                    "epoch": epoch,
                    "probe_name": probe_name,
                    "train_acc": float("nan"),
                    "test_acc": float("nan"),
                    "n_train": float("nan"),
                    "n_test": n_test,
                    "fit_seconds": float("nan"),
                    "status": "failed",
                    "error_message": f"Feature file missing: {e}",
                })
            wide_row = {"epoch": epoch}
            for p in ALL_PROBE_NAMES:
                wide_row[PROBE_WIDE_COLUMN[p]] = float("nan")
            all_wide_rows.append(wide_row)
            continue

        logger.info(
            f"  Features loaded: train={X_train_raw.shape}  test={X_test_raw.shape}"
        )

        # ── Scale features — fit ONLY on train, apply to test ────────
        X_train_scaled, X_test_scaled, _scaler = scale_features_no_leakage(
            X_train=X_train_raw,
            X_test=X_test_raw,
            logger=logger,
        )

        # ── RBF SVM: apply pre-computed subset indices to scaled train
        X_rbf_train_scaled = X_train_scaled[rbf_train_indices]

        # ── Run each requested probe ──────────────────────────────────
        epoch_test_accs = {}   # probe_name -> test_acc (for wide CSV)

        for probe_name in args.probes:

            if probe_name == "logistic_regression":
                fn = fit_logistic_regression
                fn_args = (X_train_scaled, train_labels, X_test_scaled, test_labels, config, seed)
                n_train_for_row = n_train_full

            elif probe_name == "linear_svm":
                X_linear_train_scaled = (
                    X_train_scaled[linear_train_indices]
                    if linear_train_indices is not None
                    else X_train_scaled
                )
                fn = fit_linear_svm
                fn_args = (X_linear_train_scaled, y_linear_train, X_test_scaled, test_labels, config, seed)
                n_train_for_row = n_linear_actual

            elif probe_name == "rbf_svm":
                fn = fit_rbf_svm
                fn_args = (X_rbf_train_scaled, y_rbf_train, X_test_scaled, test_labels, config, seed)
                n_train_for_row = n_rbf_actual

            elif probe_name == "random_forest":
                fn = fit_random_forest
                fn_args = (X_train_raw, train_labels, X_test_raw, test_labels, config, seed)
                n_train_for_row = n_train_full

            elif probe_name == "lightgbm":
                fn = fit_lightgbm
                fn_args = (X_train_raw, train_labels, X_test_raw, test_labels, config, seed)
                n_train_for_row = n_train_full

            else:
                logger.error(f"  Unknown probe name: {probe_name}")
                continue

            train_acc, test_acc, fit_seconds, status, error_message = run_single_probe(
                probe_name=probe_name,
                probe_fn=fn,
                fn_args=fn_args,
                logger=logger,
            )

            epoch_test_accs[probe_name] = test_acc

            long_row = {
                "epoch": epoch,
                "probe_name": probe_name,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "n_train": n_train_for_row,
                "n_test": n_test,
                "fit_seconds": round(fit_seconds, 3) if fit_seconds == fit_seconds else float("nan"),
                "status": status,
                "error_message": error_message,
            }
            all_long_rows.append(long_row)

        # ── Build wide row for this epoch ─────────────────────────────
        wide_row = {"epoch": epoch}
        for probe_name in ALL_PROBE_NAMES:
            if probe_name in epoch_test_accs:
                wide_row[PROBE_WIDE_COLUMN[probe_name]] = epoch_test_accs[probe_name]
            else:
                # Probe was not requested for this run
                wide_row[PROBE_WIDE_COLUMN[probe_name]] = float("nan")
        all_wide_rows.append(wide_row)

    # ------------------------------------------------------------------
    # 10. Save results
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("Saving results...")

    long_csv_path = os.path.join(results_dir, "probe_results_long.csv")
    save_long_csv(rows=all_long_rows, output_path=long_csv_path, logger=logger)

    wide_csv_path = os.path.join(results_dir, "probe_results_wide.csv")
    save_wide_csv(rows=all_wide_rows, output_path=wide_csv_path, logger=logger)

    # ------------------------------------------------------------------
    # 11. Summary
    # ------------------------------------------------------------------
    n_ok = sum(1 for r in all_long_rows if r["status"] == "ok")
    n_failed = sum(1 for r in all_long_rows if r["status"] == "failed")

    logger.info("=" * 70)
    logger.info("Probe evaluation complete.")
    logger.info(f"  Epochs processed: {len(epochs_to_process)}")
    logger.info(f"  Probe runs: {len(all_long_rows)}  (ok={n_ok}  failed={n_failed})")
    logger.info(f"  Long CSV:   {long_csv_path}")
    logger.info(f"  Wide CSV:   {wide_csv_path}")

    if n_failed > 0:
        logger.warning(f"  {n_failed} probe run(s) failed — check error_message column in long CSV")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
