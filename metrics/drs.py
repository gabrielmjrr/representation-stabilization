"""
metrics/drs.py — Compute temporal DRS between consecutive checkpoint epochs.

DRS (Decision Region Similarity) measures whether linear probes trained on
representations at different epochs agree on classification decisions. It
complements CKA: CKA captures geometric similarity, DRS captures functional
similarity — whether decision boundaries have changed.

Reference: Sharon & Dar (2024).

Pipeline (read top to bottom):
  1. Discover activation files; filter to DRS-eligible epochs (epoch % compute_freq == 0)
  2. Generate 500 plane triplet index sets once from the fixed seed
  3. For each eligible consecutive epoch pair (epoch_prev, epoch_curr):
       a. Reconstruct labels for the extraction set
       b. Train probe_curr on activations_curr
       c. Train probe_prev on activations_prev
       d. For each of the 500 planes:
            - Select 3 anchor points from activations_prev
            - Build a 50x50 barycentric grid in R^512
            - Apply both probes to the 2500 grid points
            - Compute fraction of points where predictions agree
       e. DRS = mean agreement fraction across all 500 planes
  4. Save results to results/drs_results.csv

Why activations_prev anchors the planes:
  probe_prev was trained on activations_prev, so the plane grid lives in
  probe_prev's training distribution. Both probes are then evaluated in this
  space, measuring whether probe_curr's decision regions match probe_prev's
  in the vicinity of the previous epoch's representations.

Usage:
    python metrics/drs.py --config configs/resnet18_cifar10.yaml

Output:
    results/drs_results.csv           — one row per computed epoch pair
    results/drs_plane_triplets.npy    — the 500 fixed index triplets (for auditability)
    results/drs.log                   — computation log
"""

import argparse
import csv
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml


# ---------------------------------------------------------------------------
# Setup utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("drs")
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
# Extraction labels — deterministic reconstruction
# ---------------------------------------------------------------------------

def reconstruct_extraction_labels(n_samples: int, n_classes: int = 10) -> np.ndarray:
    """
    Reconstruct the class labels for the fixed extraction set without reloading data.

    train.py's build_extraction_loader iterates over class indices 0..9 and appends
    n_per_class samples per class in that order. The label sequence is therefore:
        [0, 0, ..., 0, 1, 1, ..., 1, ..., 9, 9, ..., 9]
    with n_per_class = n_samples // n_classes repetitions of each class.

    The actual number of samples is n_per_class * n_classes, not n_samples, because
    integer division discards the remainder (e.g. 2048 // 10 = 204, total = 2040).

    This reconstruction is valid regardless of which specific test-set indices were
    sampled, because the CLASS ORDER is deterministic even if the within-class
    selection is random.
    """
    n_per_class = n_samples // n_classes
    labels = np.repeat(np.arange(n_classes), n_per_class)
    return labels


# ---------------------------------------------------------------------------
# Linear probe (PyTorch) — trained on frozen representations
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    """Single linear layer: maps n_features → n_classes logits."""

    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_linear_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    probe_epochs: int,
    probe_lr: float,
    batch_size: int,
    device: torch.device,
) -> LinearProbe:
    """
    Train a linear probe on frozen representations for exactly probe_epochs epochs.

    Uses Adam (not SGD) — DRS probes are internal classifiers, not the main model.
    The main training loop uses SGD; this is an isolated probe used only for DRS.

    activations: (n_samples, n_features) numpy array from one checkpoint epoch
    labels:      (n_samples,) integer class labels in [0, n_classes)

    Returns the trained probe in eval mode on device.
    """
    n_features = activations.shape[1]
    n_classes = int(labels.max()) + 1

    activations_tensor = torch.tensor(activations, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    probe_dataset = torch.utils.data.TensorDataset(activations_tensor, labels_tensor)
    probe_loader = torch.utils.data.DataLoader(
        probe_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    probe = LinearProbe(n_features=n_features, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=probe_lr)
    criterion = nn.CrossEntropyLoss()

    probe.train()
    for _ in range(probe_epochs):
        for batch_activations, batch_labels in probe_loader:
            batch_activations = batch_activations.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = probe(batch_activations)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

    probe.eval()
    return probe


# ---------------------------------------------------------------------------
# Plane triplet generation
# ---------------------------------------------------------------------------

def generate_plane_triplets(
    n_samples: int,
    n_planes: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Sample n_planes sets of 3 indices from [0, n_samples).
    Each triplet defines a 2D plane via barycentric coordinates in R^n_features.

    Returns: int64 array of shape (n_planes, 3)

    The triplets are generated once and reused for all epoch pairs. Fixing the
    triplets removes plane-sampling variance as a confound when comparing DRS
    values across different epoch pairs.
    """
    triplets = np.zeros((n_planes, 3), dtype=np.int64)
    for plane_idx in range(n_planes):
        triplets[plane_idx] = rng.choice(n_samples, size=3, replace=False)
    return triplets


# ---------------------------------------------------------------------------
# Plane grid construction
# ---------------------------------------------------------------------------

def build_plane_grid(
    anchor_a: np.ndarray,
    anchor_b: np.ndarray,
    anchor_c: np.ndarray,
    n_points_per_plane: int,
) -> np.ndarray:
    """
    Build a 2D grid of points in R^n_features via barycentric parameterization.

    The plane passing through three anchor points a, b, c is parameterized as:
        P(alpha, beta) = (1 - alpha - beta) * a + alpha * b + beta * c

    We sample a square grid of (alpha, beta) in [0, 1] x [0, 1], giving n_side^2
    grid points. Points where alpha + beta > 1 lie outside the triangle but remain
    on the same plane — including them probes decision boundaries beyond the
    immediate data support region.

    anchor_a, anchor_b, anchor_c: each shape (n_features,)
    n_points_per_plane: total grid points; must be a perfect square (e.g. 2500 = 50^2)

    Returns: grid_points of shape (n_points_per_plane, n_features)
    """
    n_side = int(np.sqrt(n_points_per_plane))
    assert n_side * n_side == n_points_per_plane, (
        f"n_points_per_plane={n_points_per_plane} must be a perfect square "
        f"(got n_side={n_side}, n_side^2={n_side ** 2})"
    )

    alpha_values = np.linspace(0.0, 1.0, n_side)   # (n_side,)
    beta_values = np.linspace(0.0, 1.0, n_side)    # (n_side,)

    alpha_grid, beta_grid = np.meshgrid(alpha_values, beta_values)   # (n_side, n_side)
    alpha_flat = alpha_grid.flatten()       # (n_points_per_plane,)
    beta_flat = beta_grid.flatten()         # (n_points_per_plane,)
    gamma_flat = 1.0 - alpha_flat - beta_flat   # (n_points_per_plane,)

    grid_points = (
        gamma_flat[:, None] * anchor_a[None, :]
        + alpha_flat[:, None] * anchor_b[None, :]
        + beta_flat[:, None] * anchor_c[None, :]
    )   # (n_points_per_plane, n_features)

    return grid_points


# ---------------------------------------------------------------------------
# DRS for one epoch pair
# ---------------------------------------------------------------------------

def compute_drs_for_pair(
    activations_curr: np.ndarray,
    activations_prev: np.ndarray,
    labels: np.ndarray,
    plane_triplets: np.ndarray,
    n_points_per_plane: int,
    probe_epochs: int,
    probe_lr: float,
    batch_size: int,
    device: torch.device,
    logger: logging.Logger,
) -> float:
    """
    Compute DRS for one (epoch_prev, epoch_curr) pair.

    Trains two probes, builds 500 plane grids anchored to activations_prev, applies
    both probes to each grid, and returns the mean agreement fraction.

    Returns: DRS value in [0, 1]. DRS=1 means both probes agree on every grid point.
    """
    logger.info("    Training probe on current epoch activations...")
    probe_curr = train_linear_probe(
        activations=activations_curr,
        labels=labels,
        probe_epochs=probe_epochs,
        probe_lr=probe_lr,
        batch_size=batch_size,
        device=device,
    )

    logger.info("    Training probe on previous epoch activations...")
    probe_prev = train_linear_probe(
        activations=activations_prev,
        labels=labels,
        probe_epochs=probe_epochs,
        probe_lr=probe_lr,
        batch_size=batch_size,
        device=device,
    )

    n_planes = plane_triplets.shape[0]
    agreement_per_plane = np.zeros(n_planes, dtype=np.float64)

    logger.info(f"    Evaluating agreement across {n_planes} planes...")
    for plane_idx in range(n_planes):
        idx_a, idx_b, idx_c = plane_triplets[plane_idx]

        # Anchor points drawn from activations_prev — the space probe_prev was trained on.
        anchor_a = activations_prev[idx_a]   # (n_features,)
        anchor_b = activations_prev[idx_b]   # (n_features,)
        anchor_c = activations_prev[idx_c]   # (n_features,)

        grid_points = build_plane_grid(
            anchor_a=anchor_a,
            anchor_b=anchor_b,
            anchor_c=anchor_c,
            n_points_per_plane=n_points_per_plane,
        )   # (n_points_per_plane, n_features)

        grid_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)

        with torch.no_grad():
            predictions_curr = probe_curr(grid_tensor).argmax(dim=1).cpu().numpy()
            predictions_prev = probe_prev(grid_tensor).argmax(dim=1).cpu().numpy()

        agreement_per_plane[plane_idx] = (predictions_curr == predictions_prev).mean()

    drs_value = float(agreement_per_plane.mean())
    return drs_value


# ---------------------------------------------------------------------------
# Main computation loop
# ---------------------------------------------------------------------------

def run_drs_computation(
    activations_dir: str,
    results_dir: str,
    config: dict,
    logger: logging.Logger,
    device: torch.device,
) -> None:
    """
    Compute DRS for every eligible consecutive checkpoint epoch pair.

    An epoch pair (epoch_prev, epoch_curr) is eligible when epoch_curr is
    divisible by drs.compute_freq. This limits computation to every compute_freq
    epochs because DRS (probe training + 500-plane evaluation) is expensive.

    Saves:
        results/drs_results.csv           — one row per computed pair
        results/drs_plane_triplets.npy    — the 500 fixed index triplets
    """
    n_classes = 10   # CIFAR-10
    n_samples_config = config["extraction"]["n_samples"]
    n_planes = config["drs"]["n_planes"]
    n_points_per_plane = config["drs"]["n_points_per_plane"]
    probe_epochs = config["drs"]["probe_epochs"]
    probe_lr = config["drs"]["probe_lr"]
    compute_freq = config["drs"]["compute_freq"]
    batch_size = config["batch_size"]
    tau = config["tau"]
    seed = config["seed"]

    logger.info(f"DRS config: n_planes={n_planes}, n_points_per_plane={n_points_per_plane}")
    logger.info(f"DRS config: probe_epochs={probe_epochs}, probe_lr={probe_lr}")
    logger.info(f"DRS config: compute_freq={compute_freq}, tau={tau}")

    labels = reconstruct_extraction_labels(n_samples=n_samples_config, n_classes=n_classes)
    n_actual_samples = labels.shape[0]
    logger.info(
        f"Extraction labels: {n_actual_samples} samples "
        f"({n_actual_samples // n_classes} per class)"
    )

    # Generate the 500 plane triplets ONCE using an isolated RNG.
    # Using a separate RandomState keeps triplet generation independent of any
    # other random operations later in this script.
    plane_rng = np.random.RandomState(seed)
    plane_triplets = generate_plane_triplets(
        n_samples=n_actual_samples,
        n_planes=n_planes,
        rng=plane_rng,
    )   # shape (n_planes, 3)

    os.makedirs(results_dir, exist_ok=True)
    plane_triplets_path = os.path.join(results_dir, "drs_plane_triplets.npy")
    np.save(plane_triplets_path, plane_triplets)
    logger.info(f"Saved plane triplets ({n_planes} triplets) to: {plane_triplets_path}")

    epoch_path_pairs = find_activation_files(activations_dir)
    n_files = len(epoch_path_pairs)
    logger.info(f"Found {n_files} activation files in '{activations_dir}'")

    if n_files < 2:
        logger.error(
            f"Need at least 2 activation files. Found only {n_files}. "
            "Run train.py first."
        )
        return

    # Filter to pairs where epoch_curr is divisible by compute_freq.
    drs_pairs = []
    for pair_idx in range(1, n_files):
        epoch_prev, path_prev = epoch_path_pairs[pair_idx - 1]
        epoch_curr, path_curr = epoch_path_pairs[pair_idx]
        if epoch_curr % compute_freq == 0:
            drs_pairs.append((epoch_prev, path_prev, epoch_curr, path_curr))

    logger.info(
        f"DRS-eligible pairs (epoch_curr % {compute_freq} == 0): {len(drs_pairs)}"
    )
    for epoch_prev, _, epoch_curr, _ in drs_pairs:
        logger.info(f"    {epoch_prev:4d} → {epoch_curr:4d}")
    logger.info("-" * 70)

    result_rows = []

    for pair_num, (epoch_prev, path_prev, epoch_curr, path_curr) in enumerate(
        drs_pairs, start=1
    ):
        logger.info(
            f"DRS pair {pair_num}/{len(drs_pairs)}: epoch {epoch_prev} → {epoch_curr}"
        )

        activations_curr = np.load(path_curr)   # (n_actual_samples, n_features)
        activations_prev = np.load(path_prev)   # (n_actual_samples, n_features)

        assert activations_curr.shape[0] == n_actual_samples, (
            f"Expected {n_actual_samples} samples in {path_curr}, "
            f"got {activations_curr.shape[0]}"
        )
        assert activations_prev.shape[0] == n_actual_samples, (
            f"Expected {n_actual_samples} samples in {path_prev}, "
            f"got {activations_prev.shape[0]}"
        )

        # Reset the seed before each pair's probe training so the probes are
        # reproducible regardless of how many pairs precede this one.
        set_seed(seed + pair_num)

        drs_value = compute_drs_for_pair(
            activations_curr=activations_curr,
            activations_prev=activations_prev,
            labels=labels,
            plane_triplets=plane_triplets,
            n_points_per_plane=n_points_per_plane,
            probe_epochs=probe_epochs,
            probe_lr=probe_lr,
            batch_size=batch_size,
            device=device,
            logger=logger,
        )

        # DRS is a similarity measure; (1 - DRS) is the change metric vs tau.
        drs_change = 1.0 - drs_value
        is_below_tau = drs_change < tau

        logger.info(
            f"  Pair {pair_num}/{len(drs_pairs)}"
            f"  epoch {epoch_prev:4d} → {epoch_curr:4d}"
            f"  DRS = {drs_value:.4f}"
            f"  change = {drs_change:.4f}"
            f"  below_tau = {'YES' if is_below_tau else 'no '}"
        )

        result_rows.append({
            "epoch_prev": epoch_prev,
            "epoch_curr": epoch_curr,
            "drs_value": round(drs_value, 6),
            "drs_change": round(drs_change, 6),
            "below_tau": int(is_below_tau),
        })

    output_path = os.path.join(results_dir, "drs_results.csv")
    fieldnames = ["epoch_prev", "epoch_curr", "drs_value", "drs_change", "below_tau"]
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    logger.info("-" * 70)
    logger.info(f"Saved DRS results ({len(result_rows)} pairs) to: {output_path}")
    logger.info(
        f"Pairs below tau={tau}: "
        f"{sum(r['below_tau'] for r in result_rows)} / {len(result_rows)}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute temporal DRS between consecutive checkpoint epochs. "
            "Run train.py first to generate activation .npy files. "
            "DRS is computed for epoch pairs where epoch_curr %% compute_freq == 0."
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
    args = parser.parse_args()

    config = load_config(args.config)

    activations_dir = args.activations_dir or config["paths"]["activations"]
    results_dir = args.results_dir or config["paths"]["results"]

    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "drs.log")
    logger = setup_logging(log_path)

    seed = config["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 70)
    logger.info("DRS Computation — Representation Stabilization")
    logger.info("=" * 70)
    logger.info(f"Config:          {args.config}")
    logger.info(f"Activations dir: {activations_dir}")
    logger.info(f"Results dir:     {results_dir}")
    logger.info(f"Device:          {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Seed: {seed}")

    run_drs_computation(
        activations_dir=activations_dir,
        results_dir=results_dir,
        config=config,
        logger=logger,
        device=device,
    )

    logger.info("=" * 70)
    logger.info("DRS computation complete.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
