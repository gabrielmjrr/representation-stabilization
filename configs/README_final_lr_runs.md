# Final LR-Sensitivity Runs

Three isolated configs for the learning-rate sensitivity study.
All runs: ResNet-18, CIFAR-10, 200 epochs, cosine annealing, SGD, seed 42.
Only `lr` and output paths differ from `resnet18_cifar10_200_fullstudy.yaml`.

| Config | lr | Output base |
|--------|-----|-------------|
| `final_lr005_seed42.yaml` | 0.05 | `/local/data/gme101/final_lr005_seed42` |
| `final_lr010_seed42.yaml` | 0.10 | `/local/data/gme101/final_lr010_seed42` |
| `final_lr020_seed42.yaml` | 0.20 | `/local/data/gme101/final_lr020_seed42` |

## Run commands (run from repo root; wrap in `nohup ... &` for remote)

```bash
# ── lr=0.05 ──────────────────────────────────────────────────────────────────
CONFIG=configs/final_lr005_seed42.yaml
python train.py                      --config $CONFIG
python features/extract_features.py  --config $CONFIG
python metrics/cka_trajectory.py     --config $CONFIG
python metrics/neural_collapse.py    --config $CONFIG
python metrics/entk_trajectory.py    --config $CONFIG
python surrogates/run_probes.py      --config $CONFIG
python analysis/build_master_table.py --config $CONFIG

# ── lr=0.10 ──────────────────────────────────────────────────────────────────
CONFIG=configs/final_lr010_seed42.yaml
python train.py                      --config $CONFIG
python features/extract_features.py  --config $CONFIG
python metrics/cka_trajectory.py     --config $CONFIG
python metrics/neural_collapse.py    --config $CONFIG
python metrics/entk_trajectory.py    --config $CONFIG
python surrogates/run_probes.py      --config $CONFIG
python analysis/build_master_table.py --config $CONFIG

# ── lr=0.20 ──────────────────────────────────────────────────────────────────
CONFIG=configs/final_lr020_seed42.yaml
python train.py                      --config $CONFIG
python features/extract_features.py  --config $CONFIG
python metrics/cka_trajectory.py     --config $CONFIG
python metrics/neural_collapse.py    --config $CONFIG
python metrics/entk_trajectory.py    --config $CONFIG
python surrogates/run_probes.py      --config $CONFIG
python analysis/build_master_table.py --config $CONFIG
```
