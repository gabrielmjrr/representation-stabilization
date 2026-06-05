"""
features/extract_features.py — Extract penultimate-layer features from saved checkpoints.

Three feature sets are produced for each checkpoint epoch:

  cka_subset  — stratified subset from the TEST set (n_per_class × 10 examples).
                Construction is identical to train.py's held-out extraction subset
                so the same seed produces the same examples and the same ordering.
                Used for CKA computation (temporal similarity between epochs).

  full_train  — all 50,000 training examples, NO augmentation, fixed ordering.
                Used for surrogate classifier training and Neural Collapse metrics
                (NC1–NC4 require per-class statistics from the training distribution).

  full_test   — all 10,000 test examples, NO augmentation, fixed ordering.
                Used for surrogate classifier evaluation.

Why no augmentation on any set:
  All three sets must produce the same feature vector for the same input image at
  every call. Augmentation is random and would break epoch-to-epoch comparability.
  model.eval() is always set before extraction.

Why fixed ordering (shuffle=False on all loaders):
  The position of a feature row in the output array must correspond to the same
  example across all epochs. Any shuffle would silently misalign features.

Pipeline (read top to bottom):
  1. Parse args and load config
  2. Determine which epochs to process (all found checkpoints, or --epochs list)
  3. Determine which feature sets to extract (default: all three)
  4. Build datasets and loaders (no augmentation)
  5. Save labels.npy and indices.npy once per split (unchanged across epochs)
  6. For each checkpoint epoch:
       a. Load model weights from checkpoint
       b. Extract features for each requested feature set
       c. Log feature shape
       d. Save features_epoch_XXXX.npy

Output (under config["paths"]["activations"]):
  cka_subset/
    labels.npy               (n_subset,)  — class labels; saved once
    indices.npy              (n_subset,)  — test-set positions of subset; saved once
    features_epoch_XXXX.npy  (n_subset, 512)
  full_train/
    labels.npy               (50000,)    — saved once
    features_epoch_XXXX.npy  (50000, 512)
  full_test/
    labels.npy               (10000,)    — saved once
    features_epoch_XXXX.npy  (10000, 512)

Usage:
  # Extract all checkpoint epochs, all three feature sets
  python features/extract_features.py --config configs/resnet18_cifar10_200_fullstudy.yaml

  # Extract only specific epochs
  python features/extract_features.py --config ... --epochs 0 5 10 100 200

  # Extract only cka_subset and full_test (skip expensive full_train)
  python features/extract_features.py --config ... --sets cka_subset full_test
"""

import argparse
import logging
import os
import sys

import numpy as np
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import yaml


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all random sources for reproducible extraction."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("extract_features")
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
# Model
# ---------------------------------------------------------------------------

def build_resnet18_cifar10(device: torch.device) -> nn.Module:
    """
    Build ResNet-18 with the CIFAR-10 stem (3×3 conv, no max-pool).
    Identical to train.py. Weights are not loaded here — call
    load_weights_from_checkpoint() to fill the model after building.
    """
    model = torchvision.models.resnet18(weights=None, num_classes=10)

    # Replace 7×7 ImageNet stem with 3×3 CIFAR-10 stem
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    # Remove max-pool (too aggressive at 32×32 spatial resolution)
    model.maxpool = nn.Identity()

    model = model.to(device)
    return model


def load_weights_from_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> int:
    """
    Load model_state_dict from a checkpoint file produced by train.py.
    Returns the epoch number stored in the checkpoint.
    Only the model weights are loaded — optimizer and scheduler are ignored.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = int(checkpoint["epoch"])
    return epoch


# ---------------------------------------------------------------------------
# Transform — shared, no augmentation
# ---------------------------------------------------------------------------

def build_no_aug_transform() -> transforms.Compose:
    """
    Deterministic preprocessing with no augmentation.
    Applied identically to all three feature sets.
    CIFAR-10 channel statistics match train.py's normalization.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])


# ---------------------------------------------------------------------------
# Dataset / loader builders
# ---------------------------------------------------------------------------

def build_cka_subset(
    n_samples: int,
    batch_size: int,
    seed: int,
):
    """
    Build the fixed stratified CKA extraction subset from the test set.

    Construction is identical to train.py's build_extraction_loader so that
    using the same seed always selects the same examples in the same order.
    This ensures that features from train.py inline extraction and features
    from this script are directly comparable.

    Why test set, not train set: the loss function directly optimises
    representations of training examples.  Measuring CKA on those same
    examples conflates genuine geometric similarity with optimisation
    pressure.  The test set is never seen during parameter updates, so
    CKA on test examples measures the network's generalised representation
    geometry without that confound.  See CLAUDE.md Activation extraction.

    n_per_class = n_samples // n_classes
    Total examples returned = n_per_class * n_classes (may be < n_samples
    when n_samples is not divisible by 10; e.g. 2048 → 204 * 10 = 2040).

    Returns:
        loader:  DataLoader (shuffle=False, fixed ordering across epochs)
        labels:  (n_selected,) int64 array of class labels for the subset
        indices: (n_selected,) int64 array of test-set positions for the subset
    """
    no_aug_transform = build_no_aug_transform()

    full_test_dataset = torchvision.datasets.CIFAR10(
        root="data/",
        train=False,
        download=True,
        transform=no_aug_transform,
    )

    n_classes = 10
    n_per_class = n_samples // n_classes

    # Group all test-set indices by class label
    indices_by_class = [[] for _ in range(n_classes)]
    for idx in range(len(full_test_dataset)):
        label = full_test_dataset.targets[idx]
        indices_by_class[label].append(idx)

    # Sample n_per_class indices from each class with a fixed RNG
    rng = np.random.RandomState(seed)
    selected_indices = []
    for class_idx in range(n_classes):
        class_indices = indices_by_class[class_idx]
        sampled = rng.choice(class_indices, size=n_per_class, replace=False)
        selected_indices.extend(sampled.tolist())

    selected_indices_array = np.array(selected_indices, dtype=np.int64)

    # Labels come from the dataset metadata — no model forward pass needed
    selected_labels = np.array(
        [full_test_dataset.targets[i] for i in selected_indices],
        dtype=np.int64,
    )

    subset_dataset = torch.utils.data.Subset(full_test_dataset, selected_indices)

    loader = torch.utils.data.DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=False,     # fixed order — same index-to-row mapping every epoch
        num_workers=4,
        pin_memory=True,
    )

    return loader, selected_labels, selected_indices_array


def build_full_train_loader(batch_size: int):
    """
    Build a DataLoader over the entire training set with NO augmentation.

    Why no augmentation: feature vectors must be deterministic so that
    the same image always maps to the same row at every checkpoint epoch.
    The training loader in train.py uses random flip + crop; this loader
    does not.  This is intentional: augmentation is for learning, not
    for representation analysis.

    Returns:
        loader: DataLoader (shuffle=False — fixed ordering across epochs)
        labels: (50000,) int64 array of class labels
    """
    no_aug_transform = build_no_aug_transform()

    train_dataset = torchvision.datasets.CIFAR10(
        root="data/",
        train=True,
        download=True,
        transform=no_aug_transform,
    )

    labels = np.array(train_dataset.targets, dtype=np.int64)

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,    # fixed ordering — same index-to-row mapping every epoch
        num_workers=4,
        pin_memory=True,
    )

    return loader, labels


def build_full_test_loader(batch_size: int):
    """
    Build a DataLoader over the entire test set with NO augmentation.

    Returns:
        loader: DataLoader (shuffle=False)
        labels: (10000,) int64 array of class labels
    """
    no_aug_transform = build_no_aug_transform()

    test_dataset = torchvision.datasets.CIFAR10(
        root="data/",
        train=False,
        download=True,
        transform=no_aug_transform,
    )

    labels = np.array(test_dataset.targets, dtype=np.int64)

    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return loader, labels


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Extract penultimate-layer features for every example in loader.

    Uses a forward hook on model.avgpool (output shape: (B, 512, 1, 1)).
    Spatial dims are squeezed to give (B, 512) per batch.

    model.eval() is set before extraction.  No gradients are computed.

    Returns:
        features: (n_samples, 512) float32 ndarray
    """
    captured = {}

    def hook_fn(module, module_input, module_output):
        # module_output: (B, 512, 1, 1) after adaptive average pooling
        captured["penultimate"] = module_output.detach().cpu()

    hook_handle = model.avgpool.register_forward_hook(hook_fn)

    model.eval()
    all_features = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            _ = model(images)    # forward pass triggers the hook

            batch_features = captured["penultimate"]                   # (B, 512, 1, 1)
            batch_features_flat = batch_features.squeeze(-1).squeeze(-1)  # (B, 512)
            all_features.append(batch_features_flat.numpy())

    hook_handle.remove()

    features = np.concatenate(all_features, axis=0)
    return features


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_checkpoint_epochs(checkpoint_dir: str) -> list:
    """
    Discover all checkpoint_epoch_XXXX.pt files in checkpoint_dir.
    Returns a list of epoch numbers sorted ascending.
    Filename convention matches train.py: checkpoint_epoch_{epoch:04d}.pt
    """
    epochs = []
    for filename in os.listdir(checkpoint_dir):
        if not filename.startswith("checkpoint_epoch_"):
            continue
        if not filename.endswith(".pt"):
            continue
        epoch_str = filename.replace("checkpoint_epoch_", "").replace(".pt", "")
        epochs.append(int(epoch_str))
    epochs.sort()
    return epochs


# ---------------------------------------------------------------------------
# Save utilities
# ---------------------------------------------------------------------------

def save_labels_and_indices(
    output_dir: str,
    labels: np.ndarray,
    indices: np.ndarray,
    logger: logging.Logger,
) -> None:
    """
    Save labels.npy and (optionally) indices.npy to output_dir.

    Called once before the epoch loop because labels and indices never change
    across checkpoints.  Passing indices=None skips writing indices.npy.
    """
    os.makedirs(output_dir, exist_ok=True)

    labels_path = os.path.join(output_dir, "labels.npy")
    np.save(labels_path, labels)
    logger.info(f"    labels.npy  shape={labels.shape}  saved: {labels_path}")

    if indices is not None:
        indices_path = os.path.join(output_dir, "indices.npy")
        np.save(indices_path, indices)
        logger.info(f"    indices.npy shape={indices.shape}  saved: {indices_path}")


def save_epoch_features(
    features: np.ndarray,
    epoch: int,
    output_dir: str,
) -> str:
    """
    Save (n_samples, 512) feature array as features_epoch_XXXX.npy.
    Returns the full path of the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"features_epoch_{epoch:04d}.npy"
    path = os.path.join(output_dir, filename)
    np.save(path, features)
    return path


# ---------------------------------------------------------------------------
# Label alignment verification
# ---------------------------------------------------------------------------

def verify_label_alignment(
    extracted_labels: np.ndarray,
    reference_labels: np.ndarray,
    set_name: str,
    epoch: int,
    logger: logging.Logger,
) -> None:
    """
    Verify that labels collected during extraction match the reference labels.

    The DataLoader must return examples in the same order as the labels array
    built from the dataset metadata.  This check confirms that no silent
    reordering occurred (e.g. from a non-deterministic DataLoader).
    """
    n_mismatches = int(np.sum(extracted_labels != reference_labels))
    if n_mismatches > 0:
        logger.error(
            f"LABEL MISMATCH in {set_name} at epoch {epoch}: "
            f"{n_mismatches} / {len(reference_labels)} labels do not match. "
            f"This indicates a DataLoader ordering bug — do NOT use these features."
        )
        raise RuntimeError(
            f"Label alignment check failed for {set_name} epoch {epoch}. "
            f"See log for details."
        )


def extract_features_with_labels(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple:
    """
    Extract features AND collect labels from loader in one forward pass.
    Used to verify that the loader ordering matches the saved labels.npy.

    Returns:
        features: (n_samples, 512) float32 ndarray
        labels:   (n_samples,)     int64 ndarray collected from the loader
    """
    captured = {}

    def hook_fn(module, module_input, module_output):
        captured["penultimate"] = module_output.detach().cpu()

    hook_handle = model.avgpool.register_forward_hook(hook_fn)

    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            _ = model(images)

            batch_features = captured["penultimate"]
            batch_features_flat = batch_features.squeeze(-1).squeeze(-1)
            all_features.append(batch_features_flat.numpy())
            all_labels.append(labels.numpy())

    hook_handle.remove()

    features = np.concatenate(all_features, axis=0)
    labels_collected = np.concatenate(all_labels, axis=0).astype(np.int64)
    return features, labels_collected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Parse arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Extract penultimate-layer features from ResNet-18 checkpoints. "
            "Run train.py first to produce checkpoint files."
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
            "Specific epoch numbers to process (e.g. --epochs 0 5 10 100 200). "
            "Default: all checkpoint files found in checkpoint_dir."
        ),
    )
    parser.add_argument(
        "--sets",
        type=str,
        nargs="+",
        choices=["cka_subset", "full_train", "full_test"],
        default=["cka_subset", "full_train", "full_test"],
        help=(
            "Feature sets to extract. Default: all three. "
            "Omit full_train to skip the expensive 50k-example extraction."
        ),
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
        for path_key in ["checkpoints", "activations", "results", "logs"]:
            if path_key in config["paths"]:
                config["paths"][path_key] = os.path.join(
                    config["paths"][path_key], f"seed_{args.seed}"
                )

    checkpoint_dir = config["paths"]["checkpoints"]
    activations_dir = config["paths"]["activations"]
    results_dir = config["paths"]["results"]
    logs_dir = config["paths"].get("logs", results_dir)

    # Each feature set gets its own subdirectory
    cka_subset_dir = os.path.join(activations_dir, "cka_subset")
    full_train_dir = os.path.join(activations_dir, "full_train")
    full_test_dir  = os.path.join(activations_dir, "full_test")

    os.makedirs(activations_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Logging
    # ------------------------------------------------------------------
    log_path = os.path.join(logs_dir, "extract_features.log")
    logger = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("Feature Extraction — Representation Stabilization")
    logger.info("=" * 70)
    logger.info(f"Config:          {args.config}")
    logger.info(f"Checkpoint dir:  {checkpoint_dir}")
    logger.info(f"Activations dir: {activations_dir}")
    logger.info(f"Sets to extract: {args.sets}")

    # ------------------------------------------------------------------
    # 4. Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # 5. Reproducibility
    # ------------------------------------------------------------------
    seed = config["seed"]
    set_seed(seed)
    logger.info(f"Seed: {seed}")

    # ------------------------------------------------------------------
    # 6. Determine which epochs to process
    # ------------------------------------------------------------------
    if args.epochs is not None:
        epochs_to_process = sorted(args.epochs)
        logger.info(f"Epochs (from --epochs): {epochs_to_process}")
    else:
        epochs_to_process = find_checkpoint_epochs(checkpoint_dir)
        logger.info(
            f"Epochs (discovered in checkpoint_dir): "
            f"{len(epochs_to_process)} checkpoints found"
        )
        logger.info(f"  Epoch list: {epochs_to_process}")

    if len(epochs_to_process) == 0:
        logger.error(
            f"No checkpoints found in '{checkpoint_dir}'. "
            "Run train.py first."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 7. Build datasets and loaders
    # ------------------------------------------------------------------
    batch_size = config["batch_size"]
    n_cka_samples = config["cka"]["n_samples"]   # e.g. 2048

    logger.info("Building datasets (no augmentation)...")

    if "cka_subset" in args.sets:
        cka_loader, cka_labels, cka_indices = build_cka_subset(
            n_samples=n_cka_samples,
            batch_size=batch_size,
            seed=seed,
        )
        n_cka_actual = len(cka_indices)
        logger.info(
            f"  cka_subset:  {n_cka_actual} examples "
            f"({n_cka_samples // 10} per class × 10 classes)"
        )

    if "full_train" in args.sets:
        full_train_loader, full_train_labels = build_full_train_loader(
            batch_size=batch_size,
        )
        logger.info(f"  full_train:  {len(full_train_labels)} examples")

    if "full_test" in args.sets:
        full_test_loader, full_test_labels = build_full_test_loader(
            batch_size=batch_size,
        )
        logger.info(f"  full_test:   {len(full_test_labels)} examples")

    # ------------------------------------------------------------------
    # 8. Save labels and indices once (unchanged across epochs)
    # ------------------------------------------------------------------
    logger.info("Saving labels and indices (once, not per epoch)...")

    if "cka_subset" in args.sets:
        logger.info("  [cka_subset]")
        save_labels_and_indices(
            output_dir=cka_subset_dir,
            labels=cka_labels,
            indices=cka_indices,
            logger=logger,
        )

    if "full_train" in args.sets:
        logger.info("  [full_train]")
        save_labels_and_indices(
            output_dir=full_train_dir,
            labels=full_train_labels,
            indices=None,     # no subset indices for the full set
            logger=logger,
        )

    if "full_test" in args.sets:
        logger.info("  [full_test]")
        save_labels_and_indices(
            output_dir=full_test_dir,
            labels=full_test_labels,
            indices=None,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # 9. Build model (weights will be loaded per epoch)
    # ------------------------------------------------------------------
    logger.info("Building ResNet-18 architecture (CIFAR-10 stem)...")
    model = build_resnet18_cifar10(device=device)

    # ------------------------------------------------------------------
    # 10. Extract features for each checkpoint epoch
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info(f"Extracting features for {len(epochs_to_process)} epochs...")
    logger.info("=" * 70)

    for epoch_idx, epoch in enumerate(epochs_to_process):
        checkpoint_filename = f"checkpoint_epoch_{epoch:04d}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

        if not os.path.exists(checkpoint_path):
            logger.warning(
                f"Checkpoint not found, skipping epoch {epoch}: {checkpoint_path}"
            )
            continue

        logger.info(
            f"Epoch {epoch:4d}  [{epoch_idx + 1}/{len(epochs_to_process)}]  "
            f"loading: {checkpoint_filename}"
        )

        loaded_epoch = load_weights_from_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            device=device,
        )

        # Sanity check: filename epoch must match stored epoch field
        if loaded_epoch != epoch:
            logger.warning(
                f"  Epoch mismatch: filename says {epoch}, "
                f"checkpoint['epoch'] says {loaded_epoch}. "
                f"Using filename epoch {epoch}."
            )

        # --- cka_subset ---
        if "cka_subset" in args.sets:
            cka_features, cka_labels_extracted = extract_features_with_labels(
                model=model,
                loader=cka_loader,
                device=device,
            )
            verify_label_alignment(
                extracted_labels=cka_labels_extracted,
                reference_labels=cka_labels,
                set_name="cka_subset",
                epoch=epoch,
                logger=logger,
            )
            cka_path = save_epoch_features(
                features=cka_features,
                epoch=epoch,
                output_dir=cka_subset_dir,
            )
            logger.info(
                f"  [cka_subset]  shape={cka_features.shape}"
                f"  dtype={cka_features.dtype}  saved: {cka_path}"
            )

        # --- full_train ---
        if "full_train" in args.sets:
            train_features, train_labels_extracted = extract_features_with_labels(
                model=model,
                loader=full_train_loader,
                device=device,
            )
            verify_label_alignment(
                extracted_labels=train_labels_extracted,
                reference_labels=full_train_labels,
                set_name="full_train",
                epoch=epoch,
                logger=logger,
            )
            train_path = save_epoch_features(
                features=train_features,
                epoch=epoch,
                output_dir=full_train_dir,
            )
            logger.info(
                f"  [full_train]  shape={train_features.shape}"
                f"  dtype={train_features.dtype}  saved: {train_path}"
            )

        # --- full_test ---
        if "full_test" in args.sets:
            test_features, test_labels_extracted = extract_features_with_labels(
                model=model,
                loader=full_test_loader,
                device=device,
            )
            verify_label_alignment(
                extracted_labels=test_labels_extracted,
                reference_labels=full_test_labels,
                set_name="full_test",
                epoch=epoch,
                logger=logger,
            )
            test_path = save_epoch_features(
                features=test_features,
                epoch=epoch,
                output_dir=full_test_dir,
            )
            logger.info(
                f"  [full_test]   shape={test_features.shape}"
                f"  dtype={test_features.dtype}  saved: {test_path}"
            )

    # ------------------------------------------------------------------
    # 11. Done
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("Feature extraction complete.")
    logger.info(f"Processed epochs: {epochs_to_process}")
    logger.info(f"Features saved to: {activations_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
