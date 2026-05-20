#!/usr/bin/env bash
# run_all_seeds.sh — Run the full experiment pipeline for multiple seeds sequentially.
#
# For each seed: train → CKA → DRS → stabilization → extract at t* → surrogates.
# If t* is not detected for a seed, the extract + surrogate steps are skipped and
# a warning is logged; the remaining seeds still run.
#
# Usage (survives terminal disconnect via nohup):
#   nohup bash run_all_seeds.sh > run_all_seeds.log 2>&1 &
#   tail -f run_all_seeds.log        # monitor progress from another terminal
#
# Or to run interactively:
#   bash run_all_seeds.sh

conda activate /local/data/gme101/thesis_env/

CONFIG="configs/resnet18_cifar10.yaml"
SEEDS="42 0 1 2 3"
BASE_RESULTS="/local/data/gme101/results"

echo "============================================================"
echo "Multi-seed pipeline"
echo "Seeds:  ${SEEDS}"
echo "Config: ${CONFIG}"
echo "Start:  $(date)"
echo "============================================================"

for SEED in ${SEEDS}; do

    echo ""
    echo "============================================================"
    echo "SEED ${SEED} — START: $(date)"
    echo "============================================================"

    # ------------------------------------------------------------------
    # Step 1: Train ResNet-18
    # Saves checkpoints and small held-out activations every 5 epochs.
    # Output: checkpoints/seed_{SEED}/ and activations/seed_{SEED}/
    # ------------------------------------------------------------------
    echo "[seed=${SEED}] Step 1/6: train.py"
    python train.py --config "${CONFIG}" --seed "${SEED}"
    if [ $? -ne 0 ]; then
        echo "[seed=${SEED}] ERROR: train.py failed — skipping this seed."
        continue
    fi

    # ------------------------------------------------------------------
    # Step 2: Compute temporal CKA between consecutive checkpoint pairs
    # Output: results/seed_{SEED}/cka_results.csv
    # ------------------------------------------------------------------
    echo "[seed=${SEED}] Step 2/6: metrics/cka.py (consecutive)"
    python metrics/cka.py --config "${CONFIG}" --seed "${SEED}"
    if [ $? -ne 0 ]; then
        echo "[seed=${SEED}] ERROR: cka.py failed — skipping this seed."
        continue
    fi

    # ------------------------------------------------------------------
    # Step 3: Compute DRS between eligible consecutive checkpoint pairs
    # Output: results/seed_{SEED}/drs_results.csv
    # Note: this is the most expensive step (~hours for 300 epochs).
    # ------------------------------------------------------------------
    echo "[seed=${SEED}] Step 3/6: metrics/drs.py"
    python metrics/drs.py --config "${CONFIG}" --seed "${SEED}"
    if [ $? -ne 0 ]; then
        echo "[seed=${SEED}] ERROR: drs.py failed — skipping this seed."
        continue
    fi

    # ------------------------------------------------------------------
    # Step 4: Detect t* from CKA + DRS curves using the (tau, K) criterion
    # Output: results/seed_{SEED}/stabilization_summary.csv
    # ------------------------------------------------------------------
    echo "[seed=${SEED}] Step 4/6: metrics/stabilization.py"
    python metrics/stabilization.py --config "${CONFIG}" --seed "${SEED}"
    if [ $? -ne 0 ]; then
        echo "[seed=${SEED}] ERROR: stabilization.py failed — skipping this seed."
        continue
    fi

    # ------------------------------------------------------------------
    # Read t* from stabilization_summary.csv.
    # The JOINT row holds max(t*_CKA, t*_DRS). If not detected the value
    # is the string NOT_DETECTED and surrogate steps are skipped.
    # ------------------------------------------------------------------
    SUMMARY_CSV="${BASE_RESULTS}/seed_${SEED}/stabilization_summary.csv"

    T_STAR=$(python -c "
import csv, sys
path = sys.argv[1]
try:
    with open(path) as f:
        for row in csv.DictReader(f):
            if row['metric'] == 'JOINT':
                val = row['t_star']
                print(val if str(val).isdigit() else 'NOT_DETECTED')
                sys.exit(0)
except Exception as e:
    pass
print('NOT_DETECTED')
" "$SUMMARY_CSV")

    echo "[seed=${SEED}] Detected t* = ${T_STAR}"

    if [ "${T_STAR}" = "NOT_DETECTED" ]; then
        echo "[seed=${SEED}] WARNING: t* not detected — skipping extract and surrogates."
        echo "[seed=${SEED}] Check ${SUMMARY_CSV} and the stabilization log for details."
        continue
    fi

    # ------------------------------------------------------------------
    # Step 5: Extract full-set activations (50 000 train + 10 000 test) at t*
    # Output: activations/seed_{SEED}/train_full_epoch_{T_STAR}.npy etc.
    # ------------------------------------------------------------------
    echo "[seed=${SEED}] Step 5/6: extract.py --epoch ${T_STAR}"
    python extract.py --config "${CONFIG}" --seed "${SEED}" --epoch "${T_STAR}"
    if [ $? -ne 0 ]; then
        echo "[seed=${SEED}] ERROR: extract.py failed — skipping surrogates."
        continue
    fi

    # ------------------------------------------------------------------
    # Step 6: Train and evaluate surrogate classifiers on frozen features at t*
    # Output: results/seed_{SEED}/surrogate_*_epoch_{T_STAR}.csv
    # ------------------------------------------------------------------
    echo "[seed=${SEED}] Step 6/6: surrogates"

    python surrogates/linear_probe.py   --config "${CONFIG}" --seed "${SEED}" --epoch "${T_STAR}"
    if [ $? -ne 0 ]; then
        echo "[seed=${SEED}] ERROR: linear_probe.py failed."
    fi

    python surrogates/lightgbm_probe.py --config "${CONFIG}" --seed "${SEED}" --epoch "${T_STAR}"
    if [ $? -ne 0 ]; then
        echo "[seed=${SEED}] ERROR: lightgbm_probe.py failed."
    fi

    python surrogates/rf_probe.py       --config "${CONFIG}" --seed "${SEED}" --epoch "${T_STAR}"
    if [ $? -ne 0 ]; then
        echo "[seed=${SEED}] ERROR: rf_probe.py failed."
    fi

    echo "[seed=${SEED}] DONE: $(date)"

done

echo ""
echo "============================================================"
echo "All seeds complete: $(date)"
echo "============================================================"
