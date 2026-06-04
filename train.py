"""
train.py — Train ResNet-18 on CIFAR-10 and save checkpoints at configured epochs.

Pipeline (read top to bottom):
  1. Parse args and load config
  2. Set seed for reproducibility
  3. Build data loaders (train + test + fixed held-out extraction set)
  4. Build model and optimizer
  5. Epoch 0: evaluate initial weights, save checkpoint + activations
  6. Train loop:
       - Forward/backward/update each batch
       - Log epoch metrics (loss, train acc, test acc)
       - Append every epoch to results/train_metrics.csv
       - Save checkpoint at configured epochs (explicit list or checkpoint_freq)
       - Extract penultimate activations at each checkpoint and save them
  7. Save run manifest (config path, git hash, hostname, CUDA info, timing)

Checkpoint epochs are controlled by config:
  checkpoint_epochs: [0, 1, 2, ...]   — explicit list (new configs)
  checkpoint_freq: N                  — every N epochs (backwards compat)
Epoch 0 and the final epoch are always checkpointed.

What is NOT done here:
  - CKA / DRS computation  (metrics/ scripts)
  - Surrogate classifier training  (surrogates/ scripts)
  - Stabilization detection  (metrics/stabilization.py)

These are deliberately separated so each step can be inspected and re-run
independently without re-running training.
"""

import argparse
import csv
import json
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

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
    """Fix all random sources so runs are fully reproducible."""
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
    """
    Write to both stdout and a log file.
    One logger is shared across the whole script.
    """
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
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


def get_checkpoint_epoch_set(config: dict, total_epochs: int) -> set:
    """
    Return the set of epoch numbers at which checkpoints will be saved.

    If config contains 'checkpoint_epochs' (explicit list), use that.
    Otherwise fall back to 'checkpoint_freq' for backwards compatibility
    with older configs that do not carry an explicit epoch list.

    Epoch 0 (initial weights, before any training) and the final epoch are
    always included regardless of which mode is used. Epoch 0 captures the
    random-initialization representation, which is the CKA baseline.
    """
    if "checkpoint_epochs" in config:
        epoch_set = set(config["checkpoint_epochs"])
    else:
        # Backwards compatibility: derive epochs from a uniform frequency
        freq = config["checkpoint_freq"]
        epoch_set = set(range(freq, total_epochs + 1, freq))

    # Always include the initialization state and the final epoch
    epoch_set.add(0)
    epoch_set.add(total_epochs)
    return epoch_set


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_cifar10_train_loader(batch_size: int, seed: int) -> torch.utils.data.DataLoader:
    """
    Standard CIFAR-10 training augmentation:
    random horizontal flip + random crop with padding.
    These are the same augmentations used in the original ResNet paper.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        # CIFAR-10 channel statistics
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="data/",
        train=True,
        download=True,
        transform=train_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        # Fix worker seed so data order is reproducible
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),
    )

    return train_loader


def build_cifar10_test_loader(batch_size: int) -> torch.utils.data.DataLoader:
    """No augmentation on test set — deterministic evaluation."""
    test_transform = transforms.Compose([
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
        transform=test_transform,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return test_loader


def build_extraction_loader(
    n_samples: int,
    batch_size: int,
    seed: int,
) -> torch.utils.data.DataLoader:
    """
    Build a fixed held-out subset of the TEST set for activation extraction.

    Why test set, not train set:
      Using training examples to extract activations that feed into surrogate
      training would leak label information. The test set is held out from
      the model's own training, making it a clean probe set.

    Why fixed:
      The same n_samples examples must be used at every checkpoint so that
      CKA compares representations of identical inputs across epochs.
      Changing the examples would confound temporal metric changes with
      sample-set changes.

    Why balanced:
      CIFAR-10 is balanced (1000 test examples per class). We sample
      n_samples // 10 examples per class to maintain this balance.
    """
    # No augmentation — deterministic representations
    extraction_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])

    full_test_dataset = torchvision.datasets.CIFAR10(
        root="data/",
        train=False,
        download=True,
        transform=extraction_transform,
    )

    n_classes = 10
    n_per_class = n_samples // n_classes

    # Group test indices by class label
    indices_by_class = [[] for _ in range(n_classes)]
    for idx, (_, label) in enumerate(full_test_dataset):
        indices_by_class[label].append(idx)

    # Sample n_per_class indices from each class with a fixed RNG
    rng = np.random.RandomState(seed)
    selected_indices = []
    for class_idx in range(n_classes):
        class_indices = indices_by_class[class_idx]
        sampled = rng.choice(class_indices, size=n_per_class, replace=False)
        selected_indices.extend(sampled.tolist())

    extraction_subset = torch.utils.data.Subset(full_test_dataset, selected_indices)

    extraction_loader = torch.utils.data.DataLoader(
        extraction_subset,
        batch_size=batch_size,
        shuffle=False,     # fixed order — critical for reproducibility
        num_workers=4,
        pin_memory=True,
    )

    return extraction_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_resnet18_cifar10(device: torch.device) -> nn.Module:
    """
    Build ResNet-18 for CIFAR-10.

    torchvision's ResNet-18 is designed for ImageNet (224×224 inputs).
    CIFAR-10 images are 32×32, so we replace the first conv layer:
      - Original: kernel_size=7, stride=2, padding=3 → severely downsamples 32×32
      - CIFAR-10: kernel_size=3, stride=1, padding=1 → preserves spatial resolution

    We also remove the first max-pool layer (another ImageNet-specific downsampling step).

    This is the standard CIFAR-10 ResNet-18 modification; see He et al. (2016).
    The penultimate layer (layer4 output, before the final FC) produces 512-dim features.
    """
    model = torchvision.models.resnet18(weights=None, num_classes=10)

    # Replace the 7×7 ImageNet stem with a 3×3 CIFAR-10 stem
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    # Remove the max-pool layer that follows conv1 in the ImageNet architecture.
    # At 32×32, this pooling would reduce spatial size to 8×8 before any residual
    # blocks — too aggressive. Replace with identity to preserve spatial resolution.
    model.maxpool = nn.Identity()

    model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# Optimizer and scheduler
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    SGD with momentum and L2 weight decay.

    Why SGD (not Adam): Sharon & Dar (2024) show that Adam produces noisier
    phase structure in CKA heatmaps and causes first-layer drift late in training
    due to gradient magnitudes falling below Adam's epsilon. SGD gives cleaner,
    more abrupt phase transitions, making t* more identifiable.
    """
    optimizer_name = config["optimizer"]
    if optimizer_name != "sgd":
        raise ValueError(
            f"Primary experiment requires SGD. Got '{optimizer_name}'. "
            "Adam is reserved for secondary/ablation runs."
        )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        nesterov=False,
    )
    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build the LR scheduler specified by config['lr_scheduler'].

    Supported values:
      cosine   — CosineAnnealingLR over the full training run (primary experiment)
      constant — no decay; LambdaLR with factor 1.0 throughout (ablation)
      step     — MultiStepLR; milestones and gamma read from config (ablation)
    """
    scheduler_name = config["lr_scheduler"]

    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["epochs"],
            eta_min=0.0,
        )

    elif scheduler_name == "constant":
        # No LR decay. LambdaLR with factor 1.0 is a no-op that keeps the
        # scheduler interface uniform (step/get_last_lr/state_dict all work).
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1.0,
        )

    elif scheduler_name == "step":
        # Multiply LR by gamma at each milestone epoch.
        # milestones and gamma must be present in config for this scheduler.
        milestones = config["lr_scheduler_milestones"]
        gamma = config["lr_scheduler_gamma"]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
        )

    else:
        raise ValueError(f"Unsupported lr_scheduler: '{scheduler_name}'")

    return scheduler


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def extract_penultimate_activations(
    model: nn.Module,
    extraction_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Extract penultimate-layer (layer4 output, before the final FC) activations
    from the fixed held-out set.

    Uses a forward hook so we do not need to modify the model architecture.
    The hook captures the output of model.layer4, which is the representation
    the network hands to its own classifier — the natural measurement point.

    Returns:
        activations: np.ndarray of shape (n_samples, 512)
    """
    captured = {}

    def hook_fn(module, module_input, module_output):
        # module_output shape: (batch, 512, 1, 1) after adaptive avg pool
        # We capture AFTER the adaptive average pool, which collapses spatial dims.
        captured["penultimate"] = module_output.detach().cpu()

    # Register hook on the adaptive average pool (output is (B, 512, 1, 1))
    # then we flatten to (B, 512) during collection
    hook_handle = model.avgpool.register_forward_hook(hook_fn)

    model.eval()
    all_activations = []

    with torch.no_grad():
        for images, _ in extraction_loader:
            images = images.to(device)
            _ = model(images)  # forward pass triggers the hook

            # captured["penultimate"] shape: (batch, 512, 1, 1)
            batch_activations = captured["penultimate"]
            # Flatten spatial dims: (batch, 512, 1, 1) → (batch, 512)
            batch_activations_flat = batch_activations.squeeze(-1).squeeze(-1)
            all_activations.append(batch_activations_flat.numpy())

    hook_handle.remove()
    model.train()

    activations = np.concatenate(all_activations, axis=0)
    return activations


# ---------------------------------------------------------------------------
# Train and eval passes
# ---------------------------------------------------------------------------

def run_train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one full pass over the training set.

    Returns:
        mean_loss: average cross-entropy loss over all batches
        accuracy:  fraction of training examples correctly classified
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item() * images.size(0)
        predicted_classes = logits.argmax(dim=1)
        total_correct += (predicted_classes == labels).sum().item()
        total_examples += images.size(0)

    mean_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    return mean_loss, accuracy


def run_eval_epoch(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate on the test set.

    Returns:
        mean_loss: average cross-entropy loss
        accuracy:  fraction of test examples correctly classified
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            predicted_classes = logits.argmax(dim=1)
            total_correct += (predicted_classes == labels).sum().item()
            total_examples += images.size(0)

    mean_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    return mean_loss, accuracy


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    learning_rate: float,
    train_loss: float,
    train_acc: float,
    test_loss: float,
    test_acc: float,
    checkpoint_dir: str,
) -> str:
    """
    Save model weights + optimizer + scheduler state so training can be resumed
    and so extract.py can load any checkpoint independently.

    Filename convention: checkpoint_epoch_{epoch:04d}.pt
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_filename = f"checkpoint_epoch_{epoch:04d}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    checkpoint = {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def save_activations(
    activations: np.ndarray,
    epoch: int,
    activations_dir: str,
) -> str:
    """
    Save extracted penultimate activations for this checkpoint epoch.

    Saved as a .npy file so extract.py / metrics scripts can load them
    without reloading and re-running the full model.

    Filename convention: activations_epoch_{epoch:04d}.npy
    """
    os.makedirs(activations_dir, exist_ok=True)

    activations_filename = f"activations_epoch_{epoch:04d}.npy"
    activations_path = os.path.join(activations_dir, activations_filename)

    np.save(activations_path, activations)
    return activations_path


# ---------------------------------------------------------------------------
# Metrics CSV
# ---------------------------------------------------------------------------

def write_metrics_csv_header(csv_path: str) -> None:
    """Create train_metrics.csv and write the column header row."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "train_acc", "test_loss", "test_acc"])


def append_metrics_csv_row(
    csv_path: str,
    epoch: int,
    lr: float,
    train_loss: float,
    train_acc: float,
    test_loss: float,
    test_acc: float,
) -> None:
    """Append one epoch's metrics as a single row to train_metrics.csv."""
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, lr, train_loss, train_acc, test_loss, test_acc])


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------

def collect_system_info(config_path: str, device: torch.device) -> dict:
    """
    Collect environment metadata for the run manifest.

    The git commit hash is recorded so any saved result can be traced back
    to the exact code that produced it.
    """
    try:
        git_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        git_commit = git_result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = "unavailable"

    system_info = {
        "config_path": os.path.abspath(config_path),
        "git_commit": git_commit,
        "hostname": socket.gethostname(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        system_info["cuda_device"] = torch.cuda.get_device_name(0)
        system_info["cuda_version"] = torch.version.cuda
        system_info["vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1
        )

    return system_info


def save_run_manifest(manifest_path: str, manifest_data: dict) -> None:
    """Write the run manifest as a JSON file."""
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Parse arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Train ResNet-18 on CIFAR-10 and save checkpoints."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/resnet18_cifar10.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint .pt file to resume training from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override the seed from config. Output files are saved to "
            "seed-specific subdirectories (e.g. checkpoints/seed_42/)."
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 2. Load config
    # ------------------------------------------------------------------
    config = load_config(args.config)

    # --seed overrides the random seed and routes all output directories to a
    # seed-specific subdirectory so concurrent seed runs don't collide.
    if args.seed is not None:
        config["seed"] = args.seed
        for path_key in ["checkpoints", "activations", "results", "logs", "manifests"]:
            if path_key in config["paths"]:
                config["paths"][path_key] = os.path.join(
                    config["paths"][path_key], f"seed_{args.seed}"
                )

    checkpoint_dir = config["paths"]["checkpoints"]
    activations_dir = config["paths"]["activations"]
    results_dir = config["paths"]["results"]
    # logs/ and manifests/ fall back to results_dir for configs that predate these keys
    logs_dir = config["paths"].get("logs", results_dir)
    manifests_dir = config["paths"].get("manifests", results_dir)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(activations_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(manifests_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Set up logging
    # ------------------------------------------------------------------
    log_path = os.path.join(logs_dir, "train.log")
    logger = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("Representation Stabilization — Training Run")
    logger.info("=" * 70)
    logger.info(f"Config: {args.config}")
    logger.info(f"Config contents:\n{yaml.dump(config, default_flow_style=False)}")

    # ------------------------------------------------------------------
    # 4. Device info
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    logger.info(f"PyTorch version: {torch.__version__}")

    # ------------------------------------------------------------------
    # 4b. Run metadata
    # ------------------------------------------------------------------
    run_start_time = datetime.now(timezone.utc)
    system_info = collect_system_info(config_path=args.config, device=device)
    logger.info(f"Hostname: {system_info['hostname']}")
    logger.info(f"Git commit: {system_info['git_commit']}")

    total_epochs = config["epochs"]
    checkpoint_epoch_set = get_checkpoint_epoch_set(config, total_epochs)
    logger.info(
        f"Checkpoint epochs ({len(checkpoint_epoch_set)} total): "
        f"{sorted(checkpoint_epoch_set)}"
    )

    # Metrics CSV: create with header if starting fresh; append rows if resuming.
    csv_path = os.path.join(results_dir, "train_metrics.csv")
    if not os.path.exists(csv_path):
        write_metrics_csv_header(csv_path)

    # Manifest path: one file per run, named by start timestamp.
    run_timestamp = run_start_time.strftime("%Y%m%d_%H%M%S")
    manifest_path = os.path.join(manifests_dir, f"manifest_{run_timestamp}.json")

    # ------------------------------------------------------------------
    # 5. Reproducibility
    # ------------------------------------------------------------------
    seed = config["seed"]
    set_seed(seed)
    logger.info(f"Seed: {seed}")

    # ------------------------------------------------------------------
    # 6. Data
    # ------------------------------------------------------------------
    batch_size = config["batch_size"]
    n_extraction_samples = config["extraction"]["n_samples"]

    logger.info("Building data loaders...")
    train_loader = build_cifar10_train_loader(batch_size=batch_size, seed=seed)
    test_loader = build_cifar10_test_loader(batch_size=batch_size)
    extraction_loader = build_extraction_loader(
        n_samples=n_extraction_samples,
        batch_size=batch_size,
        seed=seed,
    )
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Test batches:  {len(test_loader)}")
    logger.info(f"Extraction samples: {n_extraction_samples}")

    # ------------------------------------------------------------------
    # 7. Model
    # ------------------------------------------------------------------
    logger.info("Building ResNet-18 (CIFAR-10 stem)...")
    model = build_resnet18_cifar10(device=device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # 8. Loss, optimizer, scheduler
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model=model, config=config)
    scheduler = build_scheduler(optimizer=optimizer, config=config)

    logger.info(f"Optimizer: SGD  lr={config['lr']}  momentum={config['momentum']}  wd={config['weight_decay']}")
    logger.info(f"Scheduler: {config['lr_scheduler']}")

    # ------------------------------------------------------------------
    # 9. Optional: resume from checkpoint
    # ------------------------------------------------------------------
    start_epoch = 1

    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resuming from epoch {start_epoch}")

    # ------------------------------------------------------------------
    # 10. Epoch 0: save initial weights before any gradient steps
    # ------------------------------------------------------------------
    # Capturing the random-initialization state lets CKA measure how much
    # representations move from the very start of training.  train_loss and
    # train_acc are NaN because no training pass has been run yet.
    if start_epoch == 1:
        init_lr = optimizer.param_groups[0]["lr"]
        init_test_loss, init_test_acc = run_eval_epoch(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
        )
        logger.info(
            f"Epoch    0/{total_epochs}  [init]"
            f"  test_loss={init_test_loss:.4f}"
            f"  test_acc={init_test_acc:.4f}"
            f"  lr={init_lr:.6f}"
        )

        if 0 in checkpoint_epoch_set:
            init_checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=0,
                learning_rate=init_lr,
                train_loss=float("nan"),
                train_acc=float("nan"),
                test_loss=init_test_loss,
                test_acc=init_test_acc,
                checkpoint_dir=checkpoint_dir,
            )
            logger.info(f"  [checkpoint] Saved: {init_checkpoint_path}")

            init_activations = extract_penultimate_activations(
                model=model,
                extraction_loader=extraction_loader,
                device=device,
            )
            init_activations_path = save_activations(
                activations=init_activations,
                epoch=0,
                activations_dir=activations_dir,
            )
            logger.info(
                f"  [activations] shape={init_activations.shape}"
                f"  saved: {init_activations_path}"
            )

        # Record epoch 0 in the CSV regardless of whether a checkpoint was saved
        append_metrics_csv_row(
            csv_path=csv_path,
            epoch=0,
            lr=init_lr,
            train_loss=float("nan"),
            train_acc=float("nan"),
            test_loss=init_test_loss,
            test_acc=init_test_acc,
        )

    # ------------------------------------------------------------------
    # 11. Training loop
    # ------------------------------------------------------------------
    logger.info(f"Starting training: epochs {start_epoch} to {total_epochs}")
    logger.info("-" * 70)

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start_time = time.time()

        # --- Train ---
        train_loss, train_acc = run_train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        # --- Eval ---
        test_loss, test_acc = run_eval_epoch(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
        )

        # Advance LR schedule after each epoch
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_duration_seconds = time.time() - epoch_start_time

        logger.info(
            f"Epoch {epoch:4d}/{total_epochs}"
            f"  train_loss={train_loss:.4f}"
            f"  train_acc={train_acc:.4f}"
            f"  test_loss={test_loss:.4f}"
            f"  test_acc={test_acc:.4f}"
            f"  lr={current_lr:.6f}"
            f"  time={epoch_duration_seconds:.1f}s"
        )

        # Write every epoch to the CSV so downstream analysis has the full curve
        append_metrics_csv_row(
            csv_path=csv_path,
            epoch=epoch,
            lr=current_lr,
            train_loss=train_loss,
            train_acc=train_acc,
            test_loss=test_loss,
            test_acc=test_acc,
        )

        # --- Checkpoint + activation extraction ---
        if epoch in checkpoint_epoch_set:
            # Save model checkpoint
            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                learning_rate=current_lr,
                train_loss=train_loss,
                train_acc=train_acc,
                test_loss=test_loss,
                test_acc=test_acc,
                checkpoint_dir=checkpoint_dir,
            )
            logger.info(f"  [checkpoint] Saved: {checkpoint_path}")

            # Extract and save penultimate activations on the fixed held-out set.
            # We do this at every checkpoint so metrics/cka.py and metrics/drs.py
            # can load pre-extracted activations without re-running the model.
            activations = extract_penultimate_activations(
                model=model,
                extraction_loader=extraction_loader,
                device=device,
            )
            activations_path = save_activations(
                activations=activations,
                epoch=epoch,
                activations_dir=activations_dir,
            )
            logger.info(
                f"  [activations] shape={activations.shape}  "
                f"saved: {activations_path}"
            )

    # ------------------------------------------------------------------
    # 12. Done — write manifest
    # ------------------------------------------------------------------
    run_end_time = datetime.now(timezone.utc)
    manifest_data = {
        **system_info,
        "seed": seed,
        "total_epochs": total_epochs,
        "checkpoint_epochs": sorted(checkpoint_epoch_set),
        "start_time": run_start_time.isoformat(),
        "end_time": run_end_time.isoformat(),
        "duration_seconds": round((run_end_time - run_start_time).total_seconds(), 1),
        "final_test_acc": round(test_acc, 6),
        "paths": {
            "checkpoint_dir": checkpoint_dir,
            "activations_dir": activations_dir,
            "results_dir": results_dir,
            "logs_dir": logs_dir,
            "manifests_dir": manifests_dir,
            "csv": csv_path,
        },
    }
    save_run_manifest(manifest_path, manifest_data)

    logger.info("=" * 70)
    logger.info("Training complete.")
    logger.info(f"Final test accuracy: {test_acc:.4f}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")
    logger.info(f"Activations saved to: {activations_dir}")
    logger.info(f"Metrics CSV: {csv_path}")
    logger.info(f"Manifest: {manifest_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
