# Representation Stabilization in Deep Networks

Master's thesis codebase. Tests whether representational stability (within-network temporal
similarity) predicts representational sufficiency (surrogate classifier performance on frozen
features), using ResNet-18 trained on CIFAR-10.

## Research Question

> Once representations stop changing substantially, do they already contain essentially all
> task-relevant information?

Representation stabilization is measured with CKA. Representation sufficiency is measured
with surrogate classifiers trained on frozen penultimate-layer features.

---

## Storage Warning

**`/local/data` is node-local scratch.** It is wiped when the cluster job ends or the node
is reallocated. All heavy artifacts (checkpoints, feature tensors, NTK matrices) live under
`/local/data/gme101/` for performance, but they are ephemeral.

Run `scripts/archive_run_outputs.py` before your job ends to copy CSVs and plots to
persistent storage. Never commit `.pt` or `.npy` files to git.

---

## Project Structure

```
thesis/
├── train.py                        # Train ResNet-18 with cosine-annealing SGD
├── features/
│   └── extract_features.py         # Extract penultimate-layer features at every checkpoint
├── metrics/
│   ├── cka_trajectory.py           # CKA (local, to-final, mean-future) at all checkpoints
│   └── neural_collapse.py          # NC1–NC4 at every checkpoint
├── surrogates/
│   └── run_probes.py               # Fit 5 surrogate classifiers on frozen features
├── analysis/
│   ├── build_master_table.py       # Merge all CSVs into master_trajectory.csv
│   └── plot_main_trajectory.py     # Generate all main trajectory plots
├── scripts/
│   └── archive_run_outputs.py      # Copy results/plots to persistent storage
├── configs/
│   └── resnet18_cifar10_200_fullstudy.yaml   # All hyperparameters — single source of truth
└── requirements.txt
```

---

## Running One LR Experiment (Fullstudy)

All commands assume you are in the repo root. Adjust `--config` if you point to a different
config file.

### Step 1 — Train

```bash
python train.py --config configs/resnet18_cifar10_200_fullstudy.yaml
```

Trains ResNet-18 with SGD + cosine annealing for 200 epochs. Saves checkpoints and
triggers feature extraction at every configured checkpoint epoch.

Outputs (on `/local/data`):
- `checkpoints/epoch_XXXX.pt`
- `activations/full_train/features_epoch_XXXX.npy`
- `activations/full_test/features_epoch_XXXX.npy`
- `results/train_metrics.csv`

### Step 2 — Extract features (if not done by train.py)

```bash
python features/extract_features.py \
    --config configs/resnet18_cifar10_200_fullstudy.yaml \
    --sets full_train full_test
```

### Step 3 — CKA trajectory

```bash
python metrics/cka_trajectory.py \
    --config configs/resnet18_cifar10_200_fullstudy.yaml
```

Outputs: `results/cka_summary.csv`, `results/cka_matrix.csv`

### Step 4 — Neural Collapse

```bash
python metrics/neural_collapse.py \
    --config configs/resnet18_cifar10_200_fullstudy.yaml
```

Output: `results/neural_collapse.csv`

### Step 5 — eNTK trajectory

```bash
python metrics/entk_trajectory.py \
    --config configs/resnet18_cifar10_200_fullstudy.yaml
```

Computes the last-layer empirical NTK at every checkpoint from pre-extracted
`full_train` features (no autograd, no Jacobians).

Output: `results/entk_summary.csv`, `results/entk_subset_indices.json`

Smoke test:
```bash
python metrics/entk_trajectory.py --config ... --epochs 0 10 50 100 200
```

### Step 6 — Surrogate probes

```bash
python surrogates/run_probes.py \
    --config configs/resnet18_cifar10_200_fullstudy.yaml
```

Fits five classifiers (Logistic Regression, Linear SVM, RBF SVM, Random Forest, LightGBM)
on frozen features at every checkpoint. Linear SVM and RBF SVM use a 10 000-example
stratified subset to keep runtimes tractable.

Outputs: `results/probe_results_long.csv`, `results/probe_results_wide.csv`

Partial runs (smoke test):
```bash
python surrogates/run_probes.py --config ... --epochs 0 10 50 100 200
python surrogates/run_probes.py --config ... --probes logistic_regression lightgbm
```

### Step 7 — Build master table

```bash
python analysis/build_master_table.py \
    --config configs/resnet18_cifar10_200_fullstudy.yaml
```

Merges all metric CSVs into `results/master_trajectory.csv`.
Can be run with partial data — missing sources produce NaN columns.

### Step 8 — Plot

```bash
python analysis/plot_main_trajectory.py \
    --config configs/resnet18_cifar10_200_fullstudy.yaml
```

Writes eight figures to `plots/`.

### Step 9 — Archive results before job ends

```bash
python scripts/archive_run_outputs.py \
    --config configs/resnet18_cifar10_200_fullstudy.yaml
```

Copies CSVs and PNGs to `~/thesis_results/fullstudy_seed42/`. Does **not** copy
checkpoints or feature tensors. Run this before the cluster job terminates or the
node is reallocated, otherwise `/local/data` is lost.

Dry-run to preview what would be copied:
```bash
python scripts/archive_run_outputs.py --config ... --dry-run
```

---

## Key Design Decisions

**SGD, not Adam** — Adam produces noisy phase structure in CKA heatmaps and causes first-layer
representations to drift late in training (Sharon & Dar 2024). SGD gives cleaner transitions.

**Penultimate layer** — We test whether the representation the network hands to its own
classifier is sufficient. That is the penultimate (pre-FC) layer.

**Fixed held-out 2040 samples for CKA** — CKA is computed on the same fixed subset at every
checkpoint. Changing the subset would confound geometric change with sampling noise.

**Surrogate subsets for SVM** — Linear SVM and RBF SVM use a fixed 10 000-example stratified
subset. Full 50 k makes runtimes prohibitive at 44+ checkpoints. Subset indices are saved
to JSON for reproducibility. Logistic Regression, Random Forest, and LightGBM use the full
training set.

**τ fixed across all conditions** — τ=0.02 is the CKA-change stabilization threshold.
It is never tuned per run. K=5 consecutive checkpoints prevents false triggers from single
noisy dips.

**Last-layer eNTK, not full-parameter NTK** — The eNTK module computes a closed-form
kernel from the final linear layer only: K[i,j] = 1[y_i==y_j] × (h_iᵀh_j + bias_flag).
This does not require autograd or Jacobian-vector products. Cross-class entries are exactly
zero by construction — a structural property of the last-layer NTK, not numerical noise.
It is a lightweight training-dynamics diagnostic comparable to CKA, not a substitute for
the full-parameter NTK. A fixed 500-example stratified subset (50/class) is used throughout;
indices are saved to `entk_subset_indices.json` and reused across all checkpoints.

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. GPU strongly recommended for training and feature extraction.

---

## Compute Environment

VU Amsterdam JupyterLab cluster (NVIDIA L4, 23 GB VRAM, 128 CPUs).
Run training from a terminal session, not a notebook cell.
All heavy I/O goes to `/local/data/gme101/` (node-local NVMe scratch).

---

## References

- Sharon & Dar (2024) — DRS metric; three-phase CKA structure in ResNet-18
- Kornblith et al. (2019) — CKA
- Papyan, Han & Donoho (2020) — Neural Collapse
- Davari et al. (2022) — CKA reliability limitations
- Van Rossem & Saxe (2024) — universal dynamics of representation learning
