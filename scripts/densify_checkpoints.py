"""
scripts/densify_checkpoints.py

Resume training from checkpoint_epoch_0130.pt and save checkpoints at the 7
missing every-5 epochs (135, 145, 155, 165, 175, 185, 195) without overwriting
any existing checkpoint.

Training runs continuously 131→200 (no epochs are skipped — the model trajectory
must be uninterrupted); checkpoints are saved only at the 7 new target epochs.
Existing checkpoints (130, 140, 150, 160, 170, 180, 190, 200) are never touched.

All heavy-artifact I/O goes to <run-dir>/checkpoints/ directly.
--seed sets the RNG only; it never modifies any file path.

Imports build_resnet18_cifar10, build_optimizer, build_scheduler,
build_cifar10_train_loader, build_cifar10_test_loader, run_train_epoch,
run_eval_epoch, save_checkpoint, load_config, and set_seed from train.py.

Usage
-----
# Step 1 — verify on one run before committing to all 15:
python scripts/densify_checkpoints.py \
    --run-dir /local/data/gme101/final_lr005_seed42 \
    --config  configs/final_lr005_seed42.yaml \
    --verify

# Step 2 — full densification, seed-42 runs:
python scripts/densify_checkpoints.py \
    --run-dir /local/data/gme101/final_lr005_seed42 \
    --config  configs/final_lr005_seed42.yaml

python scripts/densify_checkpoints.py \
    --run-dir /local/data/gme101/final_lr010_seed42 \
    --config  configs/final_lr010_seed42.yaml

python scripts/densify_checkpoints.py \
    --run-dir /local/data/gme101/final_lr020_seed42 \
    --config  configs/final_lr020_seed42.yaml

# Non-42 seeds — same LR config, --seed sets RNG only (paths come from --run-dir):
python scripts/densify_checkpoints.py \
    --run-dir /local/data/gme101/final_lr005_seed0 \
    --config  configs/final_lr005_seed42.yaml \
    --seed 0

python scripts/densify_checkpoints.py \
    --run-dir /local/data/gme101/final_lr005_seed1 \
    --config  configs/final_lr005_seed42.yaml \
    --seed 1

python scripts/densify_checkpoints.py \
    --run-dir /local/data/gme101/final_lr005_seed2 \
    --config  configs/final_lr005_seed42.yaml \
    --seed 2

python scripts/densify_checkpoints.py \
    --run-dir /local/data/gme101/final_lr005_seed3 \
    --config  configs/final_lr005_seed42.yaml \
    --seed 3

python scripts/densify_checkpoints.py \
    --run-dir /local/data/gme101/final_lr010_seed0 \
    --config  configs/final_lr010_seed42.yaml \
    --seed 0

python scripts/densify_checkpoints.py \
    --run-dir /local/data/gme101/final_lr010_seed1 \
    --config  configs/final_lr010_seed42.yaml \
    --seed 1

python scripts/densify_checkpoints.py \
    --run-dir /local/data/gme101/final_lr010_seed2 \
    --config  configs/final_lr010_seed42.yaml \
    --seed 2

python scripts/densify_checkpoints.py \
    --run-dir /local/data/gme101/final_lr010_seed3 \
    --config  configs/final_lr010_seed42.yaml \
    --seed 3

python scripts/densify_checkpoints.py \
    --run-dir /local/data/gme101/final_lr020_seed0 \
    --config  configs/final_lr020_seed42.yaml \
    --seed 0

python scripts/densify_checkpoints.py \
    --run-dir /local/data/gme101/final_lr020_seed1 \
    --config  configs/final_lr020_seed42.yaml \
    --seed 1

python scripts/densify_checkpoints.py \
    --run-dir /local/data/gme101/final_lr020_seed2 \
    --config  configs/final_lr020_seed42.yaml \
    --seed 2

python scripts/densify_checkpoints.py \
    --run-dir /local/data/gme101/final_lr020_seed3 \
    --config  configs/final_lr020_seed42.yaml \
    --seed 3

Notes
-----
* --run-dir is fully authoritative for all checkpoint paths. The config supplies
  hyperparams (lr, momentum, weight_decay, epochs, batch_size, lr_scheduler) only.
  The config's own paths.* values are never read or used.
* --seed, if supplied, overrides config["seed"] for RNG seeding (DataLoader worker
  init and set_seed call). It does NOT append /seed_N to any path — that is
  train.py's CLI behaviour and is intentionally absent here.
* Test-set eval (run_eval_epoch over all 10k test examples) is run once per saved
  epoch, not at every training epoch, to keep wall-clock cost proportional to the
  number of new checkpoints (7 per run).
* --verify mode resumes 131→140 without saving any checkpoint, then compares the
  resulting weights to the existing checkpoint_epoch_0140.pt via relative Frobenius
  norm. Expect ~1e-4; the difference arises because the full DataLoader RNG state is
  not stored in checkpoints, so augmentation/shuffle order diverges from the original.
"""

import argparse
import os
import sys

import torch

# ---------------------------------------------------------------------------
# Locate repo root and import from train.py
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from train import (
    build_resnet18_cifar10,
    build_optimizer,
    build_scheduler,
    build_cifar10_train_loader,
    build_cifar10_test_loader,
    run_train_epoch,
    run_eval_epoch,
    save_checkpoint,
    load_config,
    set_seed,
)

# ---------------------------------------------------------------------------
# Target epochs
# ---------------------------------------------------------------------------
DENSE_START = 130
DENSE_END   = 200

# The 7 new checkpoints to create: multiples of 5 in (130, 200) that are not
# already present as multiples of 10.
NEW_CHECKPOINT_EPOCHS = [135, 145, 155, 165, 175, 185, 195]
_NEW_CHECKPOINT_SET   = set(NEW_CHECKPOINT_EPOCHS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ckpt_path(checkpoint_dir: str, epoch: int) -> str:
    return os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pt")


def load_resume_checkpoint(path: str, model, optimizer, scheduler, device) -> int:
    """Load full training state from a checkpoint; return the stored epoch number."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return int(ckpt["epoch"])


def relative_frobenius_diff(state_a: dict, state_b: dict) -> float:
    """||A - B||_F / ||B||_F summed across all parameter tensors."""
    num = 0.0
    den = 0.0
    for key in state_b:
        if key not in state_a:
            continue
        a = state_a[key].float()
        b = state_b[key].float()
        num += (a - b).norm().item() ** 2
        den += b.norm().item() ** 2
    return (num ** 0.5) / (den ** 0.5 + 1e-12)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Resume from checkpoint_epoch_0130.pt and fill in the 7 missing "
            "every-5 checkpoints (135, 145, …, 195) for one run directory."
        )
    )
    parser.add_argument(
        "--run-dir", required=True,
        help=(
            "Flat run directory, e.g. /local/data/gme101/final_lr005_seed0. "
            "Checkpoints are read from and written to <run-dir>/checkpoints/ directly. "
            "This is the sole source of truth for all file paths."
        ),
    )
    parser.add_argument(
        "--config", required=True,
        help=(
            "YAML config supplying hyperparams: lr, momentum, weight_decay, "
            "epochs, batch_size, lr_scheduler. The config's paths.* values are "
            "ignored — use the per-LR committed config for each group of seeds."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help=(
            "Override config seed for RNG only (set_seed call and DataLoader "
            "worker init). Does NOT modify any file path."
        ),
    )
    parser.add_argument(
        "--verify", action="store_true",
        help=(
            "Resume epochs 131→140 without saving, then print the relative "
            "Frobenius weight difference vs the existing checkpoint_epoch_0140.pt. "
            "Run this on one seed first; expect ~1e-4."
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve hyperparams — config supplies only these, not paths
    # ------------------------------------------------------------------
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed  # RNG override only; no path changes

    seed       = config["seed"]
    batch_size = config["batch_size"]

    # ------------------------------------------------------------------
    # Checkpoint directory — authoritative from --run-dir
    # ------------------------------------------------------------------
    checkpoint_dir = os.path.join(args.run_dir, "checkpoints")
    if not os.path.isdir(checkpoint_dir):
        print(f"ERROR: checkpoint directory not found: {checkpoint_dir}", file=sys.stderr)
        sys.exit(1)

    resume_path = ckpt_path(checkpoint_dir, DENSE_START)
    if not os.path.exists(resume_path):
        print(f"ERROR: resume checkpoint not found: {resume_path}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Device, RNG, model, optimizer, scheduler
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:      {device}")
    print(f"Run dir:     {args.run_dir}")
    print(f"Config:      {args.config}")
    print(f"Seed (RNG):  {seed}")

    set_seed(seed)

    model     = build_resnet18_cifar10(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    stored_epoch = load_resume_checkpoint(resume_path, model, optimizer, scheduler, device)
    print(f"Loaded:      {resume_path}  (checkpoint epoch = {stored_epoch})")
    if stored_epoch != DENSE_START:
        print(
            f"WARNING: checkpoint['epoch']={stored_epoch} but expected {DENSE_START}. "
            "Continuing anyway.",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    criterion    = torch.nn.CrossEntropyLoss()
    train_loader = build_cifar10_train_loader(batch_size=batch_size, seed=seed)
    test_loader  = build_cifar10_test_loader(batch_size=batch_size)

    # ------------------------------------------------------------------
    # Pre-flight: show which target epochs are missing
    # ------------------------------------------------------------------
    if not args.verify:
        missing = [
            ep for ep in NEW_CHECKPOINT_EPOCHS
            if not os.path.exists(ckpt_path(checkpoint_dir, ep))
        ]
        print(f"Target epochs: {NEW_CHECKPOINT_EPOCHS}")
        print(f"Missing ({len(missing)}): {missing}")
        if not missing:
            print("Nothing to do — all target checkpoints already exist.")
            return
    else:
        print(f"[verify] Resuming epochs {DENSE_START+1}→140, comparing to checkpoint_epoch_0140.pt")

    # ------------------------------------------------------------------
    # Training loop — runs continuously 131→200 (or 131→140 in verify)
    # ------------------------------------------------------------------
    target_epoch = 140 if args.verify else DENSE_END

    for epoch in range(DENSE_START + 1, target_epoch + 1):
        lr_this_epoch = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = run_train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        print(
            f"  epoch {epoch:3d}/{target_epoch}"
            f"  train_loss={train_loss:.4f}"
            f"  train_acc={train_acc:.4f}"
            f"  lr={lr_this_epoch:.6f}"
        )

        # In verify mode, never write anything
        if args.verify:
            continue

        # Only save at the 7 target epochs
        if epoch not in _NEW_CHECKPOINT_SET:
            continue

        dest = ckpt_path(checkpoint_dir, epoch)
        if os.path.exists(dest):
            print(f"    [skip] already exists: {dest}")
            continue

        # Run test eval once per saved epoch so test_acc/test_loss are real values
        test_loss, test_acc = run_eval_epoch(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            learning_rate=lr_this_epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            test_loss=test_loss,
            test_acc=test_acc,
            checkpoint_dir=checkpoint_dir,
        )
        print(f"    test_loss={test_loss:.4f}  test_acc={test_acc:.4f}  [saved] {dest}")

    # ------------------------------------------------------------------
    # Verify: compare resumed weights at epoch 140 to the original
    # ------------------------------------------------------------------
    if args.verify:
        ref_path = ckpt_path(checkpoint_dir, 140)
        if not os.path.exists(ref_path):
            print(f"[verify] ERROR: reference checkpoint not found: {ref_path}", file=sys.stderr)
            sys.exit(1)

        ref_ckpt = torch.load(ref_path, map_location=device, weights_only=True)
        rel_diff = relative_frobenius_diff(
            model.state_dict(),
            ref_ckpt["model_state_dict"],
        )
        print(f"\n[verify] relative Frobenius diff (resumed vs original at epoch 140): {rel_diff:.2e}")
        print(f"[verify] expected: ~1e-4")
        if rel_diff > 1e-2:
            print("[verify] WARNING: diff > 1e-2 — check scheduler/optimizer restore.")
        elif rel_diff < 1e-6:
            print("[verify] NOTE: diff < 1e-6 — unexpectedly small; checkpoints may be identical.")
        else:
            print("[verify] OK")


if __name__ == "__main__":
    main()
