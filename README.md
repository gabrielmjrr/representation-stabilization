# Representation Stabilization in Deep Networks

Master's thesis codebase. Tests whether representational stability (within-network temporal
similarity) predicts representational sufficiency (surrogate classifier performance on frozen
features), using ResNet-18 trained on CIFAR-10.

## Hypothesis

Frozen features extracted at the stabilization epoch t* — detected when both CKA and DRS
stop changing between consecutive checkpoints — are sufficient for a fresh surrogate classifier
to match full-network accuracy.

---

## Project Structure

```
representation-stabilization/
│
├── train.py                      # Train ResNet-18; saves checkpoints + activations every 5 epochs
├── extract.py                    # Extract full-scale penultimate features at a given epoch
├── metrics/
│   ├── cka.py                    # CKA(t, t-1) for all checkpoint pairs → results/cka_results.csv
│   ├── drs.py                    # DRS(t, t-1) for every 10-epoch pair → results/drs_results.csv
│   └── stabilization.py         # Detect t* from metric curves; prints next commands
├── surrogates/
│   ├── linear_probe.py           # Logistic regression on frozen features
│   ├── lightgbm_probe.py         # LightGBM on frozen features
│   └── rf_probe.py               # Random Forest on frozen features
├── configs/
│   └── resnet18_cifar10.yaml     # All hyperparameters — single source of truth
├── checkpoints/                  # .gitignore — model weights per epoch
├── activations/                  # .gitignore — extracted features per epoch
├── results/                      # Plots and CSVs — committed
└── requirements.txt
```

---

## Pipeline

### 1. Train

Trains ResNet-18 with SGD on CIFAR-10. Saves a checkpoint and extracts penultimate-layer
activations on a fixed 2048-sample held-out set at every 5-epoch interval.

```bash
python train.py --config configs/resnet18_cifar10.yaml
```

Outputs: `checkpoints/epoch_{N}.pt`, `activations/activations_epoch_{N}.npy`

### 2. Compute metrics

CKA runs on the saved activations (fast). DRS trains linear probes and evaluates them on
random 2D planes through input space (slow — computed every 10 epochs by default).

```bash
python metrics/cka.py --config configs/resnet18_cifar10.yaml   # → results/cka_results.csv
python metrics/drs.py --config configs/resnet18_cifar10.yaml   # → results/drs_results.csv
```

`cka.py` can be run before training finishes; it reads whatever activations exist.
`drs.py` can also be run mid-training — both scripts tolerate a partial activations directory.

### 3. Detect t*

Reads the metric CSVs and applies the (τ, K) stabilization criterion: t* is the first epoch
where both CKA and DRS have stayed below τ=0.02 for K=5 consecutive checkpoints.

```bash
python metrics/stabilization.py --config configs/resnet18_cifar10.yaml
```

Outputs:
- `results/stabilization_cka.csv` — annotated CKA pair table with `consecutive_count` and `is_t_star_trigger`
- `results/stabilization_drs.csv` — same for DRS
- Printed recommendation block with the exact commands to run next

**CKA-only fallback**: if `drs_results.csv` does not exist, the script warns and uses CKA
alone. Useful for inspecting mid-training without blocking on the full DRS run.

**Joint detection rule**: `t* = max(t*_CKA, t*_DRS)`. The later detection wins, because
both metrics must independently satisfy the criterion. DRS runs every 10 epochs vs CKA's
every 5, so DRS will typically be the slower detector.

### 4. Extract full-scale features at t*

Replace `85` with the epoch printed by `stabilization.py`.

```bash
python extract.py --epoch 85
```

Outputs: `activations/activations_epoch_85_full.npy` (train + test, full dataset scale)

### 5. Train surrogates

All three surrogates train on the frozen features from step 4 and report test accuracy.

```bash
python surrogates/linear_probe.py   --epoch 85
python surrogates/lightgbm_probe.py --epoch 85
python surrogates/rf_probe.py       --epoch 85
```

---

## Key Design Decisions

**SGD, not Adam** — Adam produces noisy phase structure in CKA heatmaps and causes first-layer
representations to drift late in training (Sharon & Dar 2024). SGD gives cleaner, more
abrupt stabilization transitions.

**Penultimate layer** — We test whether the representation the network hands to its own
classifier is sufficient. That is the penultimate layer.

**Fixed held-out 2048 samples** — CKA is computed on the same fixed subset at every
checkpoint. Changing the subset across epochs would confound geometric change with
sampling noise.

**τ fixed across all conditions** — τ=0.02 is the similarity-change threshold. It is never
tuned per run. K=5 consecutive checkpoints prevents false triggers from single noisy dips.

**CKA + DRS together** — CKA measures geometric similarity; DRS measures whether linear
probes at consecutive epochs make the same classification decisions. High CKA can coexist
with different functional behavior (Davari et al. 2022). Concordance between both metrics
is stronger evidence of stabilization than either alone.

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. GPU recommended for training and DRS probe computation.

---

## Compute Environment

Developed for VU Amsterdam JupyterLab cluster (NVIDIA L4, 23 GB VRAM, 128 CPUs).
Run training and DRS from a terminal, not a notebook cell.

---

## References

- Sharon & Dar (2024) — DRS metric; three-phase CKA structure in ResNet-18
- Kornblith et al. (2019) — CKA
- Papyan, Han & Donoho (2020) — Neural Collapse
- Davari et al. (2022) — CKA reliability limitations
- Van Rossem & Saxe (2024) — universal dynamics of representation learning
