# Thesis Project Context

## Research Question

Does representation stabilization imply representation sufficiency?

We define representation stabilization primarily through representation similarity metrics (CKA).

We define representation sufficiency through downstream performance of surrogate models trained on frozen representations.

The central question is:

> Once representations stop changing substantially, do they already contain essentially all task-relevant information?

---

## Previous Experiment (Completed)

### Goal

Measure representation stabilization during training under different learning-rate schedules.

### Dataset

CIFAR-10

### Architecture

ResNet-18

### Metrics

* CKA
* DRS (Dynamic Representation Stability)

### Findings

Observed CKA stabilization in most schedules where the learning rate eventually decayed sufficiently.

Approximate threshold:

tau = 0.02

where:

local_cka_change = 1 - CKA(t, previous_checkpoint)

Constant learning-rate schedules often failed to cross this threshold.

Cosine and step-decay schedules typically stabilized.

A qualitative observation from the first experiment is that learning-rate decay appears strongly related to the onset of representation stabilization.

These results already exist and should not be reproduced unless needed for debugging.

---

## Current Experiment (Active)

### Goal

Measure the relationship between:

1. Representation stabilization
2. Representation sufficiency
3. Neural Collapse
4. eNTK dynamics

throughout training.

### Main Hypothesis

Representation stabilization does not necessarily imply representation sufficiency.

Alternative outcomes:

* sufficiency precedes stabilization
* stabilization precedes sufficiency
* both emerge simultaneously

All outcomes are scientifically interesting.

---

## Experimental Setup

Dataset:

* CIFAR-10

Model:

* ResNet-18

Training:

* SGD
* momentum 0.9
* weight decay 5e-4
* cosine annealing
* 200 epochs
* batch size 128

Seed:

* 42

---

## Representation Checkpoints

Epochs:

[0,1,2,3,4,5,6,7,8,9,10,
11,12,13,14,15,16,17,18,19,20,
22,24,26,28,30,
35,40,45,50,
60,70,80,90,100,
110,120,130,140,150,160,170,180,190,200]

Dense early sampling is intentional.

---

## Metrics

### CKA

Compute:

1. Local similarity

local_cka_change

= 1 - CKA(t, previous_checkpoint)

2. Similarity to final representation

cka_to_final

= CKA(t, final_epoch)

3. Mean future similarity

mean_future_cka

= average similarity to all future stored representations

tau = 0.02 is recorded but not used as a trigger.

CKA is computed on a fixed held-out/test subset to measure representation geometry away from direct training optimization. Surrogates are trained on full_train and evaluated on full_test. Therefore CKA and probe performance are complementary but not computed on exactly the same sample set.


---

### Surrogate Models

Train on frozen representations.

Models:

* Logistic Regression
* Linear SVM
* RBF SVM
* Random Forest
* LightGBM

Probe fitting occurs at every stored checkpoint.

---

### Neural Collapse

Compute:

* NC1
* NC2
* NC3
* NC4

using penultimate-layer train representations.

---

### eNTK

Use a fixed stratified subset.

Track:

* relative Frobenius change
* cosine similarity between consecutive kernels
* spectral summaries
* class-block structure

eNTK is secondary to CKA + probes.

---

## Deprecated Components

DRS is not part of the active experiment pipeline.

Keep old code for reproducibility.

Do not build new functionality around DRS.

---

## Storage Rules

Heavy artifacts:

* checkpoints
* feature tensors
* activation tensors
* NTK matrices

must live on:

/local/data/gme101/

Git should contain only:

* code
* configs
* CSV results
* logs
* manifests
* plots

Never commit large binary artifacts.

---

## Development Philosophy

Implement one chunk at a time.

Do not refactor unrelated modules.

Preserve backwards compatibility where practical.

Prefer explicit code over abstractions.

Prioritize reproducibility and experiment reliability over elegance.
