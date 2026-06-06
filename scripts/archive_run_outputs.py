"""
scripts/archive_run_outputs.py — Copy small run results to persistent storage.

/local/data is node-local scratch on the cluster: it is wiped when the job ends
or the node is reallocated.  This script copies only the lightweight result
artifacts (CSVs, PNGs) to a persistent destination so they survive after the
scratch volume disappears.

What is copied
  results/master_trajectory.csv    (always)
  results/cka_summary.csv          (always)
  results/neural_collapse.csv      (always)
  results/probe_results_wide.csv   (if present)
  results/probe_results_long.csv   (if present)
  plots/*.png                      (all PNG files in plots dir)

What is NOT copied
  Checkpoints (.pt), feature tensors (.npy), NTK matrices — too large for git
  and for ~/thesis_results.  Keep them on /local/data for the duration of the job.

Destination
  ~/thesis_results/<run_name>/results/
  ~/thesis_results/<run_name>/plots/

<run_name> defaults to the basename of config["paths"]["base"]
(e.g. "fullstudy_seed42" from "/local/data/gme101/fullstudy_seed42").

Usage
  python scripts/archive_run_outputs.py \\
      --config configs/resnet18_cifar10_200_fullstudy.yaml

  # Override destination root
  python scripts/archive_run_outputs.py \\
      --config configs/resnet18_cifar10_200_fullstudy.yaml \\
      --dest /home/gme101/safe_results

  # Override run name
  python scripts/archive_run_outputs.py \\
      --config configs/resnet18_cifar10_200_fullstudy.yaml \\
      --run-name my_experiment_name
"""

import argparse
import glob
import os
import shutil
import sys

import yaml


ALWAYS_COPY_RESULTS = [
    "master_trajectory.csv",
    "cka_summary.csv",
    "neural_collapse.csv",
]

OPTIONAL_COPY_RESULTS = [
    "probe_results_wide.csv",
    "probe_results_long.csv",
]

FORBIDDEN_EXTENSIONS = {".pt", ".npy", ".npz", ".pkl", ".h5", ".hdf5"}


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def safe_copy(src: str, dest_path: str, dry_run: bool) -> bool:
    """
    Copy src to dest_path, creating parent directories as needed.
    Returns True on success, False on error.
    """
    ext = os.path.splitext(src)[1].lower()
    if ext in FORBIDDEN_EXTENSIONS:
        print(f"  SKIP (forbidden extension {ext}): {src}", flush=True)
        return False

    if dry_run:
        print(f"  [dry-run] would copy: {src} -> {dest_path}", flush=True)
        return True

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy2(src, dest_path)
    size_kb = os.path.getsize(dest_path) / 1024
    print(f"  OK  ({size_kb:.1f} KB)  {os.path.basename(src)} -> {dest_path}", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy lightweight run results (CSVs, PNGs) to persistent storage. "
            "Does NOT copy checkpoints, .pt, or .npy files. "
            "WARNING: /local/data is node-local scratch — run this before your job ends."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/resnet18_cifar10_200_fullstudy.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=None,
        help=(
            "Destination root directory. "
            "Default: ~/thesis_results. "
            "Results are placed under <dest>/<run_name>/."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Override the run name used as the destination subdirectory. "
            "Default: basename of config['paths']['base'] "
            "(e.g. 'fullstudy_seed42')."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without actually copying.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load config and resolve paths
    # ------------------------------------------------------------------
    if not os.path.exists(args.config):
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)

    results_dir = config["paths"]["results"]
    plots_dir = config["paths"].get("plots", os.path.join(results_dir, "plots"))
    base_dir = config["paths"].get("base", results_dir)

    run_name = args.run_name or os.path.basename(base_dir.rstrip("/\\"))
    if not run_name:
        run_name = "unnamed_run"

    dest_root = os.path.expanduser(args.dest or "~/thesis_results")
    dest_run = os.path.join(dest_root, run_name)
    dest_results = os.path.join(dest_run, "results")
    dest_plots = os.path.join(dest_run, "plots")

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    print("=" * 60, flush=True)
    print("Archive Run Outputs", flush=True)
    print("=" * 60, flush=True)
    print(f"  Config:       {args.config}", flush=True)
    print(f"  Run name:     {run_name}", flush=True)
    print(f"  Source results: {results_dir}", flush=True)
    print(f"  Source plots:   {plots_dir}", flush=True)
    print(f"  Destination:    {dest_run}", flush=True)
    if args.dry_run:
        print("  MODE: dry-run (nothing will be written)", flush=True)
    print("=" * 60, flush=True)

    copied = 0
    skipped = 0
    missing = 0

    # ------------------------------------------------------------------
    # Always-copy CSVs
    # ------------------------------------------------------------------
    print("\nResults CSVs (required):", flush=True)
    for filename in ALWAYS_COPY_RESULTS:
        src = os.path.join(results_dir, filename)
        if not os.path.exists(src):
            print(f"  MISSING: {src}", flush=True)
            missing += 1
            continue
        dest_path = os.path.join(dest_results, filename)
        if safe_copy(src, dest_path, args.dry_run):
            copied += 1
        else:
            skipped += 1

    # ------------------------------------------------------------------
    # Optional CSVs
    # ------------------------------------------------------------------
    print("\nResults CSVs (optional):", flush=True)
    for filename in OPTIONAL_COPY_RESULTS:
        src = os.path.join(results_dir, filename)
        if not os.path.exists(src):
            print(f"  not present, skipping: {filename}", flush=True)
            continue
        dest_path = os.path.join(dest_results, filename)
        if safe_copy(src, dest_path, args.dry_run):
            copied += 1
        else:
            skipped += 1

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\nPlots (*.png):", flush=True)
    if not os.path.isdir(plots_dir):
        print(f"  plots directory not found: {plots_dir}", flush=True)
    else:
        png_files = sorted(glob.glob(os.path.join(plots_dir, "*.png")))
        if not png_files:
            print("  no PNG files found", flush=True)
        for src in png_files:
            dest_path = os.path.join(dest_plots, os.path.basename(src))
            if safe_copy(src, dest_path, args.dry_run):
                copied += 1
            else:
                skipped += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("Archive complete.", flush=True)
    print(f"  Copied:  {copied}", flush=True)
    print(f"  Skipped: {skipped}", flush=True)
    print(f"  Missing: {missing}", flush=True)
    if not args.dry_run and copied > 0:
        print(f"  Destination: {dest_run}", flush=True)
    print("=" * 60, flush=True)

    if missing > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
