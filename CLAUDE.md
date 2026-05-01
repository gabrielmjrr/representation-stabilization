# CLAUDE.md — Representation Stabilization in Deep Networks
## Thesis Codebase Context

This file is the authoritative context for Claude Code working in this repository.
Read it fully before writing any code, suggesting any experiment, or modifying any file.

---

## Project Overview

This codebase implements the experiments for a master's thesis on representational
stabilization in deep neural networks. The central claim is:

> **Representational stability** (detected via within-network temporal similarity metrics)
> **predicts representational sufficiency** (measured via surrogate classifier performance
> on frozen features).

This is an empirical thesis. The code must be reproducible, well-logged, and honest
about what it measures vs what it claims.

---

## The Core Experiment

### Model 1 — Early-Stop + Surrogate
1. Train ResNet-18 with SGD on CIFAR-10
2. Every 5 epochs, extract penultimate-layer activations
3. Compute CKA(t, t-1) and DRS(t, t-1) between consecutive checkpoints
4. When both metrics stay below threshold τ for K=5 consecutive checkpoints → declare stabilization epoch t*
5. Freeze the entire trunk at t*
6. Train surrogate classifiers on frozen features: linear probe, LightGBM, Random Forest

### Model 2 — Full Network Baseline
- Same architecture, same training setup, trained to full convergence
- Standard linear classification head

### The Comparison
Does surrogate accuracy at t* match full network accuracy at convergence?
This is the sufficiency test. Stability without sufficiency is still a finding.

---

## Experimental Design Decisions (and why)

### Why SGD, not Adam
Sharon & Dar (2024) show that Adam produces noisier phase structure in CKA heatmaps
and causes first-layer representations to drift toward random initialization late in
training (due to gradient magnitudes falling below Adam's ε parameter). SGD gives
cleaner, more abrupt phase transitions — making the stabilization point t* more
identifiable. Use SGD with momentum for the primary experiment.

### Why penultimate layer
We are testing whether the learned feature representation is sufficient for
classification. The penultimate layer is the representation the network hands to its
own classifier. It is the natural measurement point.

### Why τ and K together
τ is the similarity threshold below which we consider the metric "stable".
K is the patience parameter — the number of consecutive checkpoints that must
satisfy the threshold before we declare stabilization. K prevents false triggers
from single noisy dips. Default: τ=0.02, K=5. τ must be fixed across all
experimental conditions for comparability.

### Why CKA and DRS together, not CKA alone
CKA measures geometric similarity between representation spaces across epochs.
DRS measures whether linear probes trained on representations at different epochs
make the same classification decisions. These are complementary:
- CKA can be high even when functional behavior has changed (Davari et al. 2022)
- DRS is task-specific; CKA is task-agnostic
- Concordance between CKA and DRS stabilization is stronger evidence than either alone

---

## Similarity Metrics — Implementation Notes

### CKA (Centered Kernel Alignment)
- Kornblith et al. 2019
- Measures similarity between two representation matrices X ∈ R^{n×p} and Y ∈ R^{n×q}
- Uses linear kernel: K = XX^T, L = YY^T
- Centers both Gram matrices: K' = HKH, L' = HLH where H is the centering matrix
- CKA = HSIC(K', L') / sqrt(HSIC(K', K') * HSIC(L', L'))
- Invariant to orthogonal transformations and isotropic scaling
- Known limitation: can be high for functionally different representations (Davari et al. 2022)
- We compute CKA between epoch t and epoch t-1 (consecutive checkpoints, same layer)
- This is within-network temporal CKA — NOT cross-network CKA (Kapoor et al. 2025)

### DRS (Decision Region Similarity)
- Sharon & Dar (2024)
- Train a linear probe on frozen representations at epoch t → Probe_t
- Train a linear probe on frozen representations at epoch t-1 → Probe_{t-1}
- Sample N_Q=500 random 2D planes through input space (each defined by 3 training examples)
- Discretize each plane into a 2500-point grid
- DRS = fraction of grid points where Probe_t and Probe_{t-1} agree on class label
- DRS ∈ [0, 1]; higher = more similar functional behavior
- More expensive than CKA; compute every 10 epochs unless compute allows more

### Effective Rank
- Additional geometric diagnostic (not a primary stabilization detector)
- Effective rank of the activation matrix = exp(H(σ)) where H is entropy of normalized singular values
- Tracks whether the representation is expanding or compressing during training
- Connects to Ansuini et al. 2019 intrinsic dimensionality work

### Within-class variance
- Track mean within-class variance of penultimate activations across epochs
- Should collapse toward zero as Neural Collapse (NC1) approaches
- Measure alongside CKA/DRS to locate where NC begins relative to t*

### ETF alignment
- Measure alignment of class mean vectors to a simplex ETF
- Papyan et al. (2020): class means should converge to equiangular tight frame at convergence
- Compare cos similarity between actual class mean pairs vs ideal ETF
- Helps determine whether NC geometry is forming before or after t*

---

## Key Theoretical Framing

### Three Training Phases (Sharon & Dar 2024)
The CKA heatmap (epoch × epoch) should reveal these as similarity blocks:
- **Phase I**: Learning general representations — geometry changes rapidly
- **Phase II**: Memorizing atypical/mislabeled examples — further geometric change
- **Phase III**: Post-zero-training-error — representations may continue evolving

Our t* should fall within or at the end of Phase I in most settings.

### Lazy vs Rich Regime (Woodworth et al. 2020, Chizat & Bach 2019)
- **Lazy regime**: representations barely move; network behaves like a kernel machine with fixed kernel
- **Rich regime**: representations move significantly; genuine feature learning occurs
- Determined by initialization scale — standard Xavier/He initialization puts ResNet-18 in the rich regime
- Our stabilization hypothesis only makes sense in the rich regime
- NTK (Jacot et al. 2018) predicts NO stabilization transition — it predicts stability from the start
- Finding a stabilization transition is therefore evidence we are in the rich regime

### Neural Collapse (Papyan et al. 2020)
Occurs during the Terminal Phase of Training (TPT), after training error first reaches zero:
- NC1: within-class variability collapses to zero
- NC2: class means converge to simplex ETF geometry
- NC3: classifier weights align with class means
- NC4: decisions reduce to nearest class center

NC is a late-training phenomenon. Our t* likely precedes full NC.
If surrogate underperforms at t* → NC geometry may be necessary for full sufficiency.
This is still a finding, not a failure.

### Van Rossem & Saxe (2024) — Universal Dynamics
Theoretical grounding for why stabilization occurs:
- In the rich regime at small initialization, representations follow a dynamical system
  toward a fixed point determined by data structure (not random initialization)
- Fixed point: same-class representations merge, different-class representations separate
- This fixed point corresponds approximately to our stabilization epoch t*
- Limitation: theory is derived for two-point interactions; scaling to full datasets is open
- Use this as theoretical motivation, not as a quantitative prediction tool

### Stability vs Sufficiency — The Core Distinction
- **Stability**: CKA and DRS no longer change between consecutive checkpoints
- **Sufficiency**: a fresh classifier trained on frozen features matches full network performance
- Sharon & Dar test stability only. We test both, and ask whether stability predicts sufficiency.
- High stability + high sufficiency → representations are done learning, surrogates work
- High stability + low sufficiency → NC geometry still forming; later epochs matter
- Low stability + high sufficiency → instability in metric, not in functional behavior (CKA limitation)

---

## Literature — Key Papers and Their Status

### Must Engage With (direct predecessors)

**Sharon & Dar (2024)** — closest experimental predecessor
- Compute epoch×epoch CKA heatmaps per layer for ResNet-18 and ViT on CIFAR-10
- Find three training phases as similarity blocks
- Also introduce DRS as a metric
- Do NOT test surrogate performance; do not freeze and transfer
- Our contribution: the sufficiency test after detected stabilization
- Note: They use Adam; we use SGD for cleaner phase structure

**Raghu et al. (2017) — SVCCA**
- Showed bottom-up convergence: early layers stabilize before deep layers
- First systematic use of similarity metrics on training dynamics
- Weakness: SVCCA is sensitive to subspace dimensionality choice

**Kornblith et al. (2019) — CKA**
- Introduced CKA as a more robust similarity measure
- Primarily applied to static final representations, not temporal dynamics
- Known limitation: Davari et al. (2022) showed high CKA can coexist with very different
  linear probe accuracies — similarity ≠ sufficiency

**Kapoor et al. (2025) — Convergent Learning**
- Measures cross-network alignment (between two independently trained networks)
- Finds alignment crystallizes within first epoch
- IMPORTANT: this is NOT our measurement. We measure within-network temporal stability.
  Do not conflate these. They answer different questions.

**Papyan, Han, Donoho (2020) — Neural Collapse**
- Defines NC and shows it occurs after training error reaches zero
- Late-training phenomenon; our t* likely precedes it
- Measuring NC metrics alongside t* is a core experiment

### Theoretical Framing

**Jacot et al. (2018) — NTK**
- Infinite-width networks train as kernel regressors; representations fixed from initialization
- Predicts no stabilization transition — our null hypothesis in the lazy regime
- We expect to be in the rich regime where this prediction fails

**Chizat & Bach (2019) — Lazy Training**
- Lazy training occurs at large initialization or learning rate; features barely change
- Our stabilization hypothesis requires the rich regime

**Woodworth et al. (2020) — Kernel and Rich Regimes**
- Characterizes the boundary between lazy and rich regimes
- Initial weight scale is the key determinant

**Van Rossem & Saxe (2024) — Universal Dynamics**
- Derives a universal effective theory for representation learning dynamics
- Shows rich-regime fixed point depends on data structure, not random initialization
- Theoretical anchor for why representations stabilize and what the fixed point looks like
- Limitation: two-point theory; quantitative predictions don't scale directly to CIFAR-10

**Atanasov et al. (2021) — Silent Alignment**
- Even rich networks exhibit an early phase where the kernel aligns before loss drops
- Connects lazy and rich regime behavior

### Additional Literature

**Davari et al. (2022) — CKA Reliability**
- High CKA does not guarantee similarly useful features
- CKA can be manipulated without changing functional behavior
- Direct motivation for adding DRS and surrogate tests beyond CKA

**Alain & Bengio (2016) — Linear Probes**
- Foundational precedent for using linear classifiers as representation probes
- Probed depth, not time — we extend to temporal probing

**Ansuini et al. (2019) — Intrinsic Dimensionality**
- Measured intrinsic dimensionality across layers and training
- Close to our work — measure effective rank as a parallel diagnostic

---

## Experimental Grid

### Primary (run first, understand completely)
| Parameter | Value |
|-----------|-------|
| Architecture | ResNet-18 |
| Dataset | CIFAR-10 |
| Optimizer | SGD with momentum |
| Checkpoint freq | Every 5 epochs |
| Similarity metrics | CKA + DRS |
| Surrogates | Linear probe, LightGBM, Random Forest |

### Secondary (motivated by primary findings)
| Variation | Question it answers |
|-----------|-------------------|
| Adam optimizer | Does optimizer choice shift t*? (Sharon & Dar comparison) |
| CIFAR-100 | Does dataset complexity shift t*? |
| ViT-B/16 | Does architecture affect phase structure? (synchronized vs asynchronous) |
| Label noise (20%) | Do representations re-destabilize under noise? |
| Multiple seeds | Is t* stable across random initializations? |

### Ablations
- Vary τ ∈ {0.01, 0.02, 0.05} — sensitivity of t* to threshold
- Vary K ∈ {3, 5, 10} — sensitivity to patience
- Freeze at multiple epochs {t*/2, t*, 2t*} — performance as function of freeze point
- Compare linear vs nonlinear surrogates — does nonlinearity help post-stabilization?

---

## Code Structure

```
representation-stabilization/
│
├── train.py                  # Train ResNet-18, save checkpoints every 5 epochs
├── extract.py                # Load checkpoints, extract penultimate activations
├── metrics/
│   ├── cka.py                # CKA computation (linear kernel)
│   ├── drs.py                # DRS computation (probe training + plane sampling)
│   ├── effective_rank.py     # Effective rank from singular values
│   ├── neural_collapse.py    # Within-class variance + ETF alignment
│   └── stabilization.py     # Detect t* from metric curves (τ, K logic)
├── surrogates/
│   ├── linear_probe.py       # Logistic regression on frozen features
│   ├── lightgbm_probe.py     # LightGBM on frozen features
│   └── rf_probe.py           # Random Forest on frozen features
├── configs/
│   └── resnet18_cifar10.yaml # All hyperparameters — single source of truth
├── analysis/
│   ├── plot_cka_heatmap.py   # Epoch × epoch CKA heatmap (Sharon & Dar style)
│   ├── plot_stability.py     # CKA and DRS curves over training with t* marked
│   ├── plot_surrogate.py     # Surrogate vs full network accuracy comparison
│   └── plot_geometry.py      # Effective rank, within-class variance, ETF alignment
├── checkpoints/              # .gitignore — too large for git
├── activations/              # .gitignore — extracted features
├── results/                  # Plots and CSVs — commit these
├── notebooks/
│   └── exploration.ipynb     # Scratch work only
├── requirements.txt
└── README.md
```

---

## Coding Philosophy — Readability Over Cleverness

The primary consumer of this code is a researcher who needs to manually inspect,
verify, and trust every step of the pipeline. Code that is clever, compact, or
abstract is a liability. Code that is explicit, linear, and readable is an asset.

**The core rule: write code that can be read and verified top to bottom without
needing to mentally trace through abstractions.**

### Be explicit, not clever
Do not chain operations to save lines. Do not use list comprehensions for anything
non-trivial. Do not nest function calls. Break every meaningful step into its own
named variable so each line can be read and verified independently.

```python
# BAD — clever, hard to verify
cka = (torch.trace(K_centered @ L_centered) / 
       torch.sqrt(torch.trace(K_centered @ K_centered) * torch.trace(L_centered @ L_centered)))

# GOOD — explicit, each step is verifiable
hsic_kl = torch.trace(K_centered @ L_centered)
hsic_kk = torch.trace(K_centered @ K_centered)
hsic_ll = torch.trace(L_centered @ L_centered)
cka = hsic_kl / torch.sqrt(hsic_kk * hsic_ll)
```

### Name variables after what they are, not what they do
Variable names should describe the object, not the operation that produced it.
A reader should be able to understand what a variable holds without reading the
line that created it.

```python
# BAD
result = model(x).detach().cpu().numpy()
out = scaler.fit_transform(result)

# GOOD
penultimate_activations = model(x).detach().cpu().numpy()
penultimate_activations_normalized = scaler.fit_transform(penultimate_activations)
```

### One operation per line for all metric computations
Every metric (CKA, DRS, effective rank, within-class variance) must be computed
in clearly separated steps with intermediate variables saved and named. This is
not optional. It allows inspection at any intermediate step and makes bugs obvious.

### No magic numbers anywhere
Every constant must be named and sourced from the config. If a number appears
in the code without a variable name, it is a bug waiting to happen.

```python
# BAD
planes = [sample_plane(X_train) for _ in range(500)]

# GOOD
n_planes = config['drs']['n_planes']  # 500, from Sharon & Dar (2024)
planes = [sample_plane(X_train) for _ in range(n_planes)]
```

### Comment the why, not the what
The code says what it does. Comments explain why a decision was made — especially
when the decision comes from a paper, a known limitation, or a non-obvious design choice.

```python
# Extract from a fixed held-out set, not the training set.
# Using training examples would leak label information into DRS probe training.
activations = extract_activations(model, held_out_loader)

# Use the same 500 planes for all epoch pairs.
# Plane sampling has variance; fixing the planes removes this as a confound.
drs_score = compute_drs(probe_t, probe_t_prev, fixed_planes)
```

### Keep functions small and single-purpose
Each function does exactly one thing. If a function needs a long docstring to
explain what it does, it does too many things. A reader should be able to understand
a function by reading it once without jumping to other files.

### Linear scripts, not deep call stacks
The main scripts (train.py, extract.py, etc.) should read like a sequential list
of steps. Avoid deep nesting of function calls. A reader should be able to follow
the entire experiment pipeline by reading the main script top to bottom.

---

## Coding Standards

### Reproducibility — non-negotiable
Every run must be fully reproducible. At the top of every script:
```python
import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Config files
All hyperparameters live in YAML configs. No magic numbers in code.
```yaml
model: resnet18
dataset: cifar10
optimizer: sgd
lr: 0.1
momentum: 0.9
weight_decay: 5e-4
epochs: 200
checkpoint_freq: 5
tau: 0.02
K: 5
batch_size: 128
seed: 42
```

### Logging
Use Python logging, not print statements. Every run logs:
- Config used
- GPU and CUDA version
- Epoch, train loss, train accuracy, test accuracy
- CKA(t, t-1) at each checkpoint
- DRS(t, t-1) at each DRS checkpoint
- Detected t* when triggered

### Activation extraction
Use PyTorch forward hooks, not model surgery:
```python
activations = {}
def hook_fn(module, input, output):
    activations['penultimate'] = output.detach()

hook = model.layer4.register_forward_hook(hook_fn)
```
Extract on a fixed held-out set of 2048 examples (balanced across classes).
Use the same fixed examples for every checkpoint — this is critical for comparability.

### CKA computation
Compute on GPU when possible. Use unbiased HSIC estimator for small sample sizes.
Always center both Gram matrices before computing HSIC.
Report CKA values to 4 decimal places in logs.

### DRS computation
Train each linear probe for exactly 10 epochs with Adam lr=0.0001.
Use 500 random planes, 2500 points per plane.
Cache the 500 plane triplets — use the same planes for all epoch pairs.
DRS is expensive: compute every 10 epochs in the primary run.

---

## What Not To Do

- Do not use Adam for the primary experiment (first-layer artifact, noisy phases)
- Do not compute CKA on the full training set — use a fixed held-out subset
- Do not tune τ per run — fix it across all conditions
- Do not conflate within-network temporal CKA (ours) with cross-network CKA (Kapoor)
- Do not claim stabilization from CKA alone — DRS must corroborate
- Do not treat a surrogate underperformance as a failed experiment — it is a finding
- Do not run secondary experiments before the primary is fully understood
- Do not hardcode hyperparameters in scripts — everything goes in configs

---

## Open Questions (guide what to measure, not just what to run)

1. Does t* coincide with the epoch at which linear probe accuracy plateaus?
2. Does NC geometry (ETF alignment, within-class variance collapse) begin before or after t*?
3. Is t* stable across random seeds for the same architecture/dataset?
4. Does the accuracy gap between surrogate and full network correlate with distance from t*?
5. Can t* be predicted from the loss curve or gradient norms, without computing CKA/DRS?
6. Does effective rank stabilize at the same epoch as CKA/DRS?

---

## Honest Weaknesses — Acknowledge in Code Comments and Thesis

- CKA stabilization ≠ representation stabilization (measurement problem — Davari et al.)
- DRS uses linear probes; nonlinear structure may still be changing even when DRS is flat
- Stabilization criterion (τ, K) is heuristic — results will depend on these choices
- Surrogate outperforming full network could reflect classifier quality differences, not representation quality
- Two-month timeline means secondary experiments may be incomplete — be honest about scope
- Van Rossem & Saxe theory applies to two-point interactions; connection to CIFAR-10 is qualitative

---

## Compute Environment

- Cluster: VU Amsterdam JupyterLab hub
- GPU: NVIDIA L4, 23GB VRAM
- CPUs: 128
- Run long experiments via terminal, not notebook cells
- Save checkpoints to /checkpoints/, results to /results/
- Use SLURM if queue access is available for multi-run experiments
