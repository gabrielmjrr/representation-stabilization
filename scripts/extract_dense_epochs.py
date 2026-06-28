"""
scripts/extract_dense_epochs.py

Extract penultimate-layer features for the 7 new dense-window epochs
(135, 145, 155, 165, 175, 185, 195) for one run directory.

For EXISTING (densified) runs, asserts that the recomputed cka_subset indices
are byte-identical to activations/cka_subset/indices.npy BEFORE writing
any feature file. If they differ the script exits 1 and nothing is written.

For NEW runs (no indices.npy yet), the assertion is skipped and the subset
is freshly established.

All 3 feature sets are extracted for all 7 epochs.
Why not a reduced full_train set: run_extended_analysis.py discovers epochs
from cka_subset and then unconditionally loads full_train for every discovered
epoch. Partially extracting full_train (e.g. only 135,155,175,195) would cause
the analysis to fail at 145,165,185 with no graceful fallback. Extracting all 3
sets costs ~127 MB/epoch x 7 x 15 runs ≈ 13 GB — 4% of available disk.

Imports from features/extract_features.py via importlib (no __init__.py needed).
--run-dir is fully authoritative for all paths. --seed sets RNG only.

Usage
-----
python scripts/extract_dense_epochs.py \
    --run-dir /local/data/gme101/final_lr005_seed42 \
    --config  configs/final_lr005_seed42.yaml

python scripts/extract_dense_epochs.py \
    --run-dir /local/data/gme101/final_lr005_seed0 \
    --config  configs/final_lr005_seed42.yaml \
    --seed 0
"""

import argparse
import importlib.util
import logging
import os
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Repo root and imports
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from train import load_config, set_seed

# Load features/extract_features.py without requiring a features/__init__.py
_fe_spec = importlib.util.spec_from_file_location(
    "extract_features",
    os.path.join(_REPO_ROOT, "features", "extract_features.py"),
)
_fe = importlib.util.module_from_spec(_fe_spec)
_fe_spec.loader.exec_module(_fe)

# ---------------------------------------------------------------------------
# Target epochs
# ---------------------------------------------------------------------------
NEW_CHECKPOINT_EPOCHS = [135, 145, 155, 165, 175, 185, 195]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logger(name: str = "extract_dense") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
    return logger


def _assert_indices_match(indices_path: str, recomputed: np.ndarray, run_dir: str) -> None:
    """
    Load existing indices.npy and compare to recomputed.
    Called BEFORE any write — exits 1 immediately on mismatch.
    If indices.npy doesn't exist yet (new run) the check is skipped.
    """
    if not os.path.exists(indices_path):
        print("  [assertion] No existing indices.npy — new run, skipping alignment check.")
        return

    existing = np.load(indices_path)
    if not np.array_equal(existing, recomputed):
        print(f"\nFATAL: cka_subset indices MISMATCH in {run_dir}", file=sys.stderr)
        print(f"  existing  indices[:5]: {existing[:5]}", file=sys.stderr)
        print(f"  recomputed indices[:5]: {recomputed[:5]}", file=sys.stderr)
        print(
            "  This means --seed does not match the seed used in the original run.\n"
            "  No feature files have been written. Halting.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  [assertion] OK — {len(existing)} indices match saved indices.npy.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract features for the 7 new dense-window epochs for one run directory. "
            "Asserts cka_subset alignment before writing anything."
        )
    )
    parser.add_argument(
        "--run-dir", required=True,
        help=(
            "Flat run directory, e.g. /local/data/gme101/final_lr005_seed0. "
            "All paths are derived from this; config paths.* are ignored."
        ),
    )
    parser.add_argument(
        "--config", required=True,
        help=(
            "YAML config for hyperparams (batch_size, cka.n_samples, seed). "
            "Use the per-LR committed config; override seed with --seed."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override config seed for RNG only. Does NOT modify any path.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Config (hyperparams only — paths come from --run-dir)
    # ------------------------------------------------------------------
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed

    seed          = config["seed"]
    batch_size    = config["batch_size"]
    n_cka_samples = config["cka"]["n_samples"]

    # ------------------------------------------------------------------
    # Paths — all from --run-dir
    # ------------------------------------------------------------------
    checkpoint_dir = os.path.join(args.run_dir, "checkpoints")
    activations_dir = os.path.join(args.run_dir, "activations")
    cka_subset_dir  = os.path.join(activations_dir, "cka_subset")
    full_train_dir  = os.path.join(activations_dir, "full_train")
    full_test_dir   = os.path.join(activations_dir, "full_test")

    if not os.path.isdir(checkpoint_dir):
        print(f"ERROR: checkpoint directory not found: {checkpoint_dir}", file=sys.stderr)
        sys.exit(1)

    for d in [cka_subset_dir, full_train_dir, full_test_dir]:
        os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # Device and RNG
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:  {device}")
    print(f"Run dir: {args.run_dir}")
    print(f"Seed:    {seed}")
    print(f"Target epochs: {NEW_CHECKPOINT_EPOCHS}\n")

    set_seed(seed)

    # ------------------------------------------------------------------
    # Build datasets (read-only — no writes yet)
    # ------------------------------------------------------------------
    print("Building datasets...")
    cka_loader, cka_labels, cka_indices = _fe.build_cka_subset(
        n_samples=n_cka_samples, batch_size=batch_size, seed=seed,
    )
    full_train_loader, full_train_labels = _fe.build_full_train_loader(batch_size=batch_size)
    full_test_loader,  full_test_labels  = _fe.build_full_test_loader(batch_size=batch_size)

    print(
        f"  cka_subset:  {len(cka_indices)} examples  "
        f"({n_cka_samples // 10} per class × 10 classes)"
    )
    print(f"  full_train:  {len(full_train_labels)} examples")
    print(f"  full_test:   {len(full_test_labels)} examples")

    # ------------------------------------------------------------------
    # Assertion — must pass before any file writes
    # ------------------------------------------------------------------
    print("\nChecking cka_subset alignment...")
    _assert_indices_match(
        os.path.join(cka_subset_dir, "indices.npy"),
        cka_indices,
        args.run_dir,
    )

    # ------------------------------------------------------------------
    # Write labels / indices (idempotent after assertion passes)
    # ------------------------------------------------------------------
    logger = _setup_logger()
    _fe.save_labels_and_indices(cka_subset_dir, cka_labels, cka_indices, logger)
    _fe.save_labels_and_indices(full_train_dir,  full_train_labels, None,        logger)
    _fe.save_labels_and_indices(full_test_dir,   full_test_labels,  None,        logger)

    # ------------------------------------------------------------------
    # Check which target epochs have checkpoints
    # ------------------------------------------------------------------
    extractable = []
    for ep in NEW_CHECKPOINT_EPOCHS:
        cp = os.path.join(checkpoint_dir, f"checkpoint_epoch_{ep:04d}.pt")
        if os.path.exists(cp):
            extractable.append(ep)
        else:
            print(f"  WARNING: checkpoint not found for epoch {ep}, will skip: {cp}")

    if not extractable:
        print("No target checkpoints found. Exiting.")
        return

    print(f"\nExtracting features for epochs: {extractable}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Build model once; weights loaded per epoch
    # ------------------------------------------------------------------
    model = _fe.build_resnet18_cifar10(device)

    splits = [
        ("cka_subset", cka_loader,        cka_labels,        cka_subset_dir),
        ("full_train",  full_train_loader, full_train_labels, full_train_dir),
        ("full_test",   full_test_loader,  full_test_labels,  full_test_dir),
    ]

    # ------------------------------------------------------------------
    # Feature extraction loop
    # ------------------------------------------------------------------
    for epoch in extractable:
        cp = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pt")
        loaded_ep = _fe.load_weights_from_checkpoint(model, cp, device)

        if loaded_ep != epoch:
            print(
                f"  WARNING: filename says epoch {epoch}, "
                f"checkpoint['epoch']={loaded_ep}. Proceeding with filename epoch."
            )

        print(f"\nEpoch {epoch}  [checkpoint epoch = {loaded_ep}]")

        for split_name, loader, ref_labels, split_dir in splits:
            dest = os.path.join(split_dir, f"features_epoch_{epoch:04d}.npy")
            if os.path.exists(dest):
                print(f"  [skip] {split_name} already exists: {dest}")
                continue

            feats, extracted_labels = _fe.extract_features_with_labels(model, loader, device)
            _fe.verify_label_alignment(extracted_labels, ref_labels, split_name, epoch, logger)
            saved = _fe.save_epoch_features(feats, epoch, split_dir)
            print(f"  [{split_name}]  shape={feats.shape}  dtype={feats.dtype}  -> {saved}")

    print(f"\n{'='*60}\nExtraction complete for {args.run_dir}")


if __name__ == "__main__":
    main()
