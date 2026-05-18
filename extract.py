"""
extract.py — Extract penultimate-layer activations from a specific checkpoint epoch
for the full training and test sets.

Train.py already extracts activations for a small fixed held-out set (2048 samples)
at every checkpoint — those are used by CKA and DRS, which require the same examples
across all epochs for temporal comparison.

This script extracts full-scale features (50,000 train + 10,000 test) from a SINGLE
chosen checkpoint. These features are the input to the surrogate classifiers in
surrogates/. We extract on demand rather than during training because full-set
extraction at every checkpoint would be ~25x more expensive than the 2048-sample
extraction, and we only need it at t* (and optionally a few comparison epochs).

Pipeline (read top to bottom):
  1. Parse args (--epoch is required; this is the checkpoint to load)
  2. Load config and build data loaders (no augmentation, no shuffle)
  3. Load model from checkpoint at the specified epoch
  4. Extract activations + labels for the full training set → train_full_epoch_{epoch}.npy
  5. Extract activations + labels for the full test set    → test_full_epoch_{epoch}.npy
  6. Save labels once (same for all epochs)

Usage:
    python extract.py --config configs/resnet18_cifar10.yaml --epoch 85

Output:
    activations/train_full_epoch_{epoch:04d}.npy   — shape (50000, 512)
    activations/test_full_epoch_{epoch:04d}.npy    — shape (10000, 512)
    activations/train_labels.npy                   — shape (50000,) int class labels
    activations/test_labels.npy                    — shape (10000,) int class labels
    results/extract.log
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import yaml


# ---------------------------------------------------------------------------
# Setup utilities  (mirrors train.py — same seed and logging pattern)
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("extract")
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
# Data loaders — no augmentation, no shuffle (critical for saved-label alignment)
# ---------------------------------------------------------------------------

def build_full_train_loader_no_augmentation(
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """
    Full 50,000-sample training set with NO augmentation, NO shuffle.

    No augmentation: we need deterministic representations — the same input must
    always produce the same activation vector so saved features and labels align.

    No shuffle: labels are collected in the same pass and saved separately. If
    the loader shuffled, the saved labels would not correspond to the saved features.
    """
    extraction_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="data/",
        train=True,
        download=True,
        transform=extraction_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader


def build_full_test_loader(batch_size: int) -> torch.utils.data.DataLoader:
    """
    Full 10,000-sample test set. No augmentation, no shuffle.
    Same rationale as build_full_train_loader_no_augmentation.
    """
    extraction_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])

    test_dataset = torchvision.datasets.CIFAR10(
        root="data/",
        train=False,
        download=True,
        transform=extraction_transform,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return test_loader


# ---------------------------------------------------------------------------
# Model — same build function as train.py
# ---------------------------------------------------------------------------

def build_resnet18_cifar10(device: torch.device) -> nn.Module:
    """
    Build ResNet-18 with the CIFAR-10 stem modification.
    Architecture must match train.py exactly so checkpoint weights load correctly.
    """
    model = torchvision.models.resnet18(weights=None, num_classes=10)

    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    # Remove ImageNet max-pool that would collapse 32×32 inputs too aggressively.
    model.maxpool = nn.Identity()

    model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def extract_activations_and_labels(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract penultimate-layer (avgpool output) activations and class labels for
    every example in loader.

    Hook is registered on model.avgpool, which produces shape (B, 512, 1, 1).
    We flatten to (B, 512) — this is the 512-dim representation the network
    hands to its own linear classifier.

    Returns:
        activations: np.ndarray of shape (n_examples, 512)
        labels:      np.ndarray of shape (n_examples,) integer class labels in [0, 9]
    """
    captured = {}

    def hook_fn(module, module_input, module_output):
        captured["penultimate"] = module_output.detach().cpu()

    hook_handle = model.avgpool.register_forward_hook(hook_fn)

    model.eval()

    all_activations = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            _ = model(images)

            # Shape: (batch, 512, 1, 1) → (batch, 512)
            batch_activations = captured["penultimate"]
            batch_activations_flat = batch_activations.squeeze(-1).squeeze(-1)

            all_activations.append(batch_activations_flat.numpy())
            all_labels.append(labels.numpy())

    hook_handle.remove()

    activations = np.concatenate(all_activations, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)

    return activations, labels_array


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    device: torch.device,
    logger: logging.Logger,
) -> dict:
    """
    Load model weights from a checkpoint file. Returns the full checkpoint dict
    so callers can inspect saved metrics (test_acc, train_acc, etc.).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Run train.py first, or check that --epoch matches a saved checkpoint."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    logger.info(f"  Checkpoint train_acc: {checkpoint['train_acc']:.4f}")
    logger.info(f"  Checkpoint test_acc:  {checkpoint['test_acc']:.4f}")

    return checkpoint


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_activations_npy(
    activations: np.ndarray,
    split: str,
    epoch: int,
    activations_dir: str,
    logger: logging.Logger,
) -> str:
    """
    Save full-set activations for a given split (train or test) and epoch.
    Filename: {split}_full_epoch_{epoch:04d}.npy
    """
    os.makedirs(activations_dir, exist_ok=True)
    filename = f"{split}_full_epoch_{epoch:04d}.npy"
    path = os.path.join(activations_dir, filename)
    np.save(path, activations)
    logger.info(f"  Saved {split} activations: shape={activations.shape}  path={path}")
    return path


def save_labels_npy(
    labels: np.ndarray,
    split: str,
    activations_dir: str,
    logger: logging.Logger,
) -> str:
    """
    Save class labels for a split. Labels are epoch-independent — the same
    examples appear in the same order regardless of which checkpoint we extract
    from (because we use shuffle=False and the same dataset). We save once and
    re-use across epochs.

    Filename: {split}_labels.npy
    """
    os.makedirs(activations_dir, exist_ok=True)
    filename = f"{split}_labels.npy"
    path = os.path.join(activations_dir, filename)
    np.save(path, labels)
    logger.info(f"  Saved {split} labels:      shape={labels.shape}  path={path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Parse arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Extract full-set penultimate-layer activations from a checkpoint epoch. "
            "Run train.py first to create checkpoints. "
            "These features are used by surrogates/ — not by CKA or DRS."
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
        help=(
            "Checkpoint epoch to extract from (e.g. 85 for t*). "
            "Must match a saved checkpoint_epoch_XXXX.pt file."
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 2. Load config
    # ------------------------------------------------------------------
    config = load_config(args.config)

    checkpoint_dir = config["paths"]["checkpoints"]
    activations_dir = config["paths"]["activations"]
    results_dir = config["paths"]["results"]

    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Logging
    # ------------------------------------------------------------------
    log_path = os.path.join(results_dir, "extract.log")
    logger = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("Activation Extraction — Full Training and Test Sets")
    logger.info("=" * 70)
    logger.info(f"Config:          {args.config}")
    logger.info(f"Epoch to load:   {args.epoch}")
    logger.info(f"Checkpoint dir:  {checkpoint_dir}")
    logger.info(f"Activations dir: {activations_dir}")

    # ------------------------------------------------------------------
    # 4. Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    set_seed(config["seed"])

    # ------------------------------------------------------------------
    # 5. Build model and load checkpoint
    # ------------------------------------------------------------------
    logger.info("Building model...")
    model = build_resnet18_cifar10(device=device)

    checkpoint_filename = f"checkpoint_epoch_{args.epoch:04d}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    checkpoint_data = load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        device=device,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # 6. Build data loaders
    # ------------------------------------------------------------------
    batch_size = config["batch_size"]

    logger.info("Building full train loader (no augmentation, no shuffle)...")
    train_loader = build_full_train_loader_no_augmentation(batch_size=batch_size)
    logger.info(f"  Train batches: {len(train_loader)}")

    logger.info("Building full test loader...")
    test_loader = build_full_test_loader(batch_size=batch_size)
    logger.info(f"  Test batches:  {len(test_loader)}")

    # ------------------------------------------------------------------
    # 7. Extract activations for train set
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("Extracting activations for full training set...")

    train_activations, train_labels = extract_activations_and_labels(
        model=model,
        loader=train_loader,
        device=device,
    )

    save_activations_npy(
        activations=train_activations,
        split="train",
        epoch=args.epoch,
        activations_dir=activations_dir,
        logger=logger,
    )
    save_labels_npy(
        labels=train_labels,
        split="train",
        activations_dir=activations_dir,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # 8. Extract activations for test set
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("Extracting activations for full test set...")

    test_activations, test_labels = extract_activations_and_labels(
        model=model,
        loader=test_loader,
        device=device,
    )

    save_activations_npy(
        activations=test_activations,
        split="test",
        epoch=args.epoch,
        activations_dir=activations_dir,
        logger=logger,
    )
    save_labels_npy(
        labels=test_labels,
        split="test",
        activations_dir=activations_dir,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("Extraction complete.")
    logger.info(f"  Train features: {train_activations.shape}")
    logger.info(f"  Test features:  {test_activations.shape}")
    logger.info(
        f"  Checkpoint network test accuracy at epoch {args.epoch}: "
        f"{checkpoint_data['test_acc']:.4f}"
    )
    logger.info(
        "  Next step: run surrogates/linear_probe.py, lightgbm_probe.py, "
        "or rf_probe.py with --epoch same as above."
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
