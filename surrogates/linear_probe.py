"""
surrogates/linear_probe.py — Train and evaluate a logistic regression surrogate
on penultimate-layer features extracted at a specific checkpoint epoch.

The central question this script answers:
  Are the frozen features at epoch t sufficient for a fresh linear classifier
  to match the accuracy of the full trained network?

Input:
  activations/train_full_epoch_{epoch:04d}.npy   — (50000, 512) train features
  activations/train_labels.npy                   — (50000,) train labels
  activations/test_full_epoch_{epoch:04d}.npy    — (10000, 512) test features
  activations/test_labels.npy                    — (10000,) test labels
  checkpoints/checkpoint_epoch_{epoch:04d}.pt    — for the reference test_acc

These files are produced by extract.py. Run extract.py --epoch {epoch} first.

Usage:
    python surrogates/linear_probe.py --config configs/resnet18_cifar10.yaml --epoch 85

Output:
    results/surrogate_linear_epoch_{epoch:04d}.csv   — accuracy results
    results/surrogate.log                            — appended log
"""

import argparse
import csv
import logging
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Setup utilities
# ---------------------------------------------------------------------------

def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("surrogate_linear")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Append to shared surrogate.log so all surrogate runs are in one file
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def load_features_and_labels(
    activations_dir: str,
    split: str,
    epoch: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load pre-extracted activation features and labels for a given split and epoch.

    split: "train" or "test"
    epoch: the checkpoint epoch these features were extracted from

    Returns:
        features: (n_samples, 512) float32 array
        labels:   (n_samples,)     int array
    """
    features_filename = f"{split}_full_epoch_{epoch:04d}.npy"
    features_path = os.path.join(activations_dir, features_filename)

    labels_filename = f"{split}_labels.npy"
    labels_path = os.path.join(activations_dir, labels_filename)

    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"Features not found: {features_path}\n"
            f"Run: python extract.py --epoch {epoch}"
        )
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Labels not found: {labels_path}\n"
            f"Run: python extract.py --epoch {epoch}"
        )

    features = np.load(features_path)
    labels = np.load(labels_path)

    return features, labels


# ---------------------------------------------------------------------------
# Reference accuracy from checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint_test_acc(
    checkpoint_dir: str,
    epoch: int,
) -> float:
    """
    Load the test accuracy the full network achieved at this checkpoint epoch.
    Used as the reference to compare surrogate performance against.
    """
    checkpoint_filename = f"checkpoint_epoch_{epoch:04d}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Cannot retrieve reference accuracy."
        )

    # Load only the metadata fields, not the full model weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    test_acc = float(checkpoint["test_acc"])
    return test_acc


# ---------------------------------------------------------------------------
# Surrogate training and evaluation
# ---------------------------------------------------------------------------

def normalize_features(
    train_features: np.ndarray,
    test_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit StandardScaler on train features; apply to both train and test.

    Logistic regression is sensitive to feature scale. Normalizing ensures that
    the 512 activation dimensions contribute equally to the classifier and that
    the regularization strength C has a consistent meaning across epochs.

    We fit on train only — fitting on test would leak information.
    """
    scaler = StandardScaler()

    train_features_normalized = scaler.fit_transform(train_features)
    test_features_normalized = scaler.transform(test_features)

    return train_features_normalized, test_features_normalized


def train_logistic_regression(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    C: float,
    max_iter: int,
    seed: int,
) -> LogisticRegression:
    """
    Train a logistic regression (multinomial, L-BFGS) on the training features.

    C: inverse regularization strength (larger C = less regularization)
    max_iter: maximum iterations for the L-BFGS solver
    seed: for reproducibility of the solver's internal random state

    L-BFGS is preferred over SGD for logistic regression here because:
      - It is more reliable at convergence on fixed feature representations
      - It does not require learning rate tuning (unlike SGD)
      - 50,000 × 512 is small enough for full-batch optimization
    """
    clf = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=seed,
        verbose=0,
    )
    clf.fit(train_features, train_labels)
    return clf


def evaluate_classifier(
    clf: LogisticRegression,
    features: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Return fraction of correctly classified examples."""
    predictions = clf.predict(features)
    accuracy = (predictions == labels).mean()
    return float(accuracy)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Parse arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Train a logistic regression surrogate on frozen features at a checkpoint "
            "epoch and compare its accuracy to the full network at the same epoch. "
            "Run extract.py --epoch {epoch} first."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/resnet18_cifar10.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        required=True,
        help="Checkpoint epoch whose features to evaluate (must match an extract.py run).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 2. Load config
    # ------------------------------------------------------------------
    config = load_config(args.config)

    activations_dir = config["paths"]["activations"]
    checkpoint_dir = config["paths"]["checkpoints"]
    results_dir = config["paths"]["results"]
    seed = config["seed"]

    surrogate_C = config["surrogates"]["linear"]["C"]
    surrogate_max_iter = config["surrogates"]["linear"]["max_iter"]

    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Logging
    # ------------------------------------------------------------------
    log_path = os.path.join(results_dir, "surrogate.log")
    logger = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("Surrogate: Logistic Regression (Linear Probe)")
    logger.info("=" * 70)
    logger.info(f"Config:          {args.config}")
    logger.info(f"Epoch:           {args.epoch}")
    logger.info(f"C:               {surrogate_C}")
    logger.info(f"max_iter:        {surrogate_max_iter}")

    # ------------------------------------------------------------------
    # 4. Load features
    # ------------------------------------------------------------------
    logger.info("Loading train features...")
    train_features, train_labels = load_features_and_labels(
        activations_dir=activations_dir,
        split="train",
        epoch=args.epoch,
    )
    logger.info(f"  Train: features={train_features.shape}  labels={train_labels.shape}")

    logger.info("Loading test features...")
    test_features, test_labels = load_features_and_labels(
        activations_dir=activations_dir,
        split="test",
        epoch=args.epoch,
    )
    logger.info(f"  Test:  features={test_features.shape}  labels={test_labels.shape}")

    # ------------------------------------------------------------------
    # 5. Normalize features
    # ------------------------------------------------------------------
    logger.info("Normalizing features (StandardScaler fit on train)...")
    train_features_normalized, test_features_normalized = normalize_features(
        train_features=train_features,
        test_features=test_features,
    )

    # ------------------------------------------------------------------
    # 6. Load reference accuracy from checkpoint
    # ------------------------------------------------------------------
    network_test_acc = load_checkpoint_test_acc(
        checkpoint_dir=checkpoint_dir,
        epoch=args.epoch,
    )
    logger.info(f"Reference (full network) test_acc at epoch {args.epoch}: {network_test_acc:.4f}")

    # ------------------------------------------------------------------
    # 7. Train logistic regression
    # ------------------------------------------------------------------
    logger.info("Training logistic regression...")
    clf = train_logistic_regression(
        train_features=train_features_normalized,
        train_labels=train_labels,
        C=surrogate_C,
        max_iter=surrogate_max_iter,
        seed=seed,
    )
    n_iterations = clf.n_iter_[0]
    logger.info(f"  Solver converged in {n_iterations} iterations.")

    # ------------------------------------------------------------------
    # 8. Evaluate
    # ------------------------------------------------------------------
    surrogate_train_acc = evaluate_classifier(
        clf=clf,
        features=train_features_normalized,
        labels=train_labels,
    )
    surrogate_test_acc = evaluate_classifier(
        clf=clf,
        features=test_features_normalized,
        labels=test_labels,
    )
    accuracy_gap = network_test_acc - surrogate_test_acc

    logger.info("-" * 70)
    logger.info(f"Surrogate train accuracy:        {surrogate_train_acc:.4f}")
    logger.info(f"Surrogate test accuracy:         {surrogate_test_acc:.4f}")
    logger.info(f"Full network test accuracy:      {network_test_acc:.4f}")
    logger.info(
        f"Accuracy gap (network - surrogate): {accuracy_gap:+.4f}"
        f"  ({'surrogate matches or exceeds' if accuracy_gap <= 0 else 'surrogate underperforms'})"
    )

    # ------------------------------------------------------------------
    # 9. Save results
    # ------------------------------------------------------------------
    output_filename = f"surrogate_linear_epoch_{args.epoch:04d}.csv"
    output_path = os.path.join(results_dir, output_filename)

    fieldnames = [
        "epoch",
        "surrogate_type",
        "surrogate_train_acc",
        "surrogate_test_acc",
        "network_test_acc",
        "accuracy_gap",
        "C",
        "n_iterations",
    ]
    row = {
        "epoch": args.epoch,
        "surrogate_type": "logistic_regression",
        "surrogate_train_acc": round(surrogate_train_acc, 6),
        "surrogate_test_acc": round(surrogate_test_acc, 6),
        "network_test_acc": round(network_test_acc, 6),
        "accuracy_gap": round(accuracy_gap, 6),
        "C": surrogate_C,
        "n_iterations": n_iterations,
    }

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    logger.info(f"Saved results to: {output_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
