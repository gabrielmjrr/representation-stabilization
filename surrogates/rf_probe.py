"""
surrogates/rf_probe.py — Train and evaluate a Random Forest surrogate on
penultimate-layer features extracted at a specific checkpoint epoch.

Random Forest is included alongside logistic regression and LightGBM to probe
whether the representation is linearly sufficient or whether nonlinear classifiers
can extract additional accuracy from features at t*. Unlike LightGBM (gradient
boosting), Random Forest trains its trees independently — this tends to be
more robust to hyperparameter choice, at the cost of some accuracy.

If RF ≈ LightGBM >> logistic regression: the representation has strong nonlinear
  structure; linear probes are not sufficient for the sufficiency test.
If all three ≈ equal: the representation has become linearly separable by t*.

Input:
  activations/train_full_epoch_{epoch:04d}.npy   — (50000, 512) train features
  activations/train_labels.npy                   — (50000,) train labels
  activations/test_full_epoch_{epoch:04d}.npy    — (10000, 512) test features
  activations/test_labels.npy                    — (10000,) test labels
  checkpoints/checkpoint_epoch_{epoch:04d}.pt    — for the reference test_acc

Run extract.py --epoch {epoch} first.

Usage:
    python surrogates/rf_probe.py --config configs/resnet18_cifar10.yaml --epoch 85

Output:
    results/surrogate_rf_epoch_{epoch:04d}.csv   — accuracy results
    results/surrogate.log                        — appended log
"""

import argparse
import csv
import logging
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------------------------------
# Setup utilities
# ---------------------------------------------------------------------------

def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("surrogate_rf")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ---------------------------------------------------------------------------
# Feature loading  (identical pattern to linear_probe.py and lightgbm_probe.py)
# ---------------------------------------------------------------------------

def load_features_and_labels(
    activations_dir: str,
    split: str,
    epoch: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load pre-extracted activation features and labels for a given split and epoch.
    Files are produced by extract.py.
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


def load_checkpoint_test_acc(checkpoint_dir: str, epoch: int) -> float:
    """Load the full network test accuracy saved at this checkpoint."""
    checkpoint_filename = f"checkpoint_epoch_{epoch:04d}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return float(checkpoint["test_acc"])


# ---------------------------------------------------------------------------
# Surrogate training and evaluation
# ---------------------------------------------------------------------------

def train_random_forest(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    n_estimators: int,
    max_features: str,
    n_jobs: int,
    seed: int,
    logger: logging.Logger,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on the training features.

    Random Forest is scale-invariant (tree splits depend on rank, not magnitude),
    so we do not normalize features before passing them in.

    n_estimators: number of trees in the forest
    max_features: features considered at each split — "sqrt" is the standard
                  for classification (sqrt of total features = sqrt(512) ≈ 22)
    n_jobs:       parallel threads for tree fitting (-1 = all available)
    seed:         controls both bootstrap sampling and feature selection per split
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        n_jobs=n_jobs,
        random_state=seed,
        verbose=0,
    )

    logger.info(
        f"  Fitting RandomForest: n_estimators={n_estimators}, "
        f"max_features={max_features}, n_jobs={n_jobs}"
    )
    clf.fit(train_features, train_labels)

    return clf


def evaluate_classifier(
    clf: RandomForestClassifier,
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
            "Train a Random Forest surrogate on frozen features at a checkpoint epoch "
            "and compare its accuracy to the full network at the same epoch. "
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
        help="Checkpoint epoch whose features to evaluate.",
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

    # ------------------------------------------------------------------
    # 2. Load config
    # ------------------------------------------------------------------
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

    activations_dir = config["paths"]["activations"]
    checkpoint_dir = config["paths"]["checkpoints"]
    results_dir = config["paths"]["results"]
    seed = config["seed"]

    n_estimators = config["surrogates"]["random_forest"]["n_estimators"]
    max_features = config["surrogates"]["random_forest"]["max_features"]
    n_jobs = config["surrogates"]["random_forest"]["n_jobs"]

    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Logging
    # ------------------------------------------------------------------
    log_path = os.path.join(results_dir, "surrogate.log")
    logger = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("Surrogate: Random Forest")
    logger.info("=" * 70)
    logger.info(f"Config:          {args.config}")
    logger.info(f"Epoch:           {args.epoch}")
    logger.info(
        f"RandomForest:    n_estimators={n_estimators}  "
        f"max_features={max_features}  n_jobs={n_jobs}"
    )

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
    # 5. Load reference accuracy
    # ------------------------------------------------------------------
    network_test_acc = load_checkpoint_test_acc(
        checkpoint_dir=checkpoint_dir,
        epoch=args.epoch,
    )
    logger.info(f"Reference (full network) test_acc at epoch {args.epoch}: {network_test_acc:.4f}")

    # ------------------------------------------------------------------
    # 6. Train Random Forest
    # ------------------------------------------------------------------
    logger.info("Training Random Forest...")
    clf = train_random_forest(
        train_features=train_features,
        train_labels=train_labels,
        n_estimators=n_estimators,
        max_features=max_features,
        n_jobs=n_jobs,
        seed=seed,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # 7. Evaluate
    # ------------------------------------------------------------------
    surrogate_train_acc = evaluate_classifier(
        clf=clf,
        features=train_features,
        labels=train_labels,
    )
    surrogate_test_acc = evaluate_classifier(
        clf=clf,
        features=test_features,
        labels=test_labels,
    )
    accuracy_gap = network_test_acc - surrogate_test_acc

    logger.info("-" * 70)
    logger.info(f"Surrogate train accuracy:           {surrogate_train_acc:.4f}")
    logger.info(f"Surrogate test accuracy:            {surrogate_test_acc:.4f}")
    logger.info(f"Full network test accuracy:         {network_test_acc:.4f}")
    logger.info(
        f"Accuracy gap (network - surrogate): {accuracy_gap:+.4f}"
        f"  ({'surrogate matches or exceeds' if accuracy_gap <= 0 else 'surrogate underperforms'})"
    )

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------
    output_filename = f"surrogate_rf_epoch_{args.epoch:04d}.csv"
    output_path = os.path.join(results_dir, output_filename)

    fieldnames = [
        "epoch",
        "surrogate_type",
        "surrogate_train_acc",
        "surrogate_test_acc",
        "network_test_acc",
        "accuracy_gap",
        "n_estimators",
        "max_features",
    ]
    row = {
        "epoch": args.epoch,
        "surrogate_type": "random_forest",
        "surrogate_train_acc": round(surrogate_train_acc, 6),
        "surrogate_test_acc": round(surrogate_test_acc, 6),
        "network_test_acc": round(network_test_acc, 6),
        "accuracy_gap": round(accuracy_gap, 6),
        "n_estimators": n_estimators,
        "max_features": max_features,
    }

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    logger.info(f"Saved results to: {output_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
