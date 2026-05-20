"""
metrics/stabilization.py — Detect t* from CKA and DRS metric curves.

Reads:
  results/cka_results.csv   — produced by metrics/cka.py
  results/drs_results.csv   — produced by metrics/drs.py (optional but recommended)

Applies the (τ, K) stabilization criterion independently to each metric:
  Scan consecutive checkpoint pairs in order. Maintain a running count of
  how many pairs in a row satisfy change < τ. The moment that count first
  reaches K, declare that pair's epoch_curr as the metric's t*.

  CKA change  = 1 − CKA(epoch_prev, epoch_curr)    — small means similar
  DRS change  = 1 − DRS(epoch_prev, epoch_curr)    — small means similar

  Any pair where change ≥ τ resets the consecutive count to zero (patience lost).

The joint t* is max(t*_CKA, t*_DRS): we require BOTH metrics to have satisfied
the criterion, and we declare stabilization at the epoch where the slower metric
finally crossed its threshold. If DRS results are absent, the script falls back
to CKA-only detection and warns.

Design note on CKA vs DRS cadence:
  CKA is computed at every 5-epoch checkpoint pair (checkpoint_freq = 5).
  DRS is computed every compute_freq epochs (default 10), so it has half as many
  pairs. K consecutive DRS pairs therefore span twice as many epochs as K CKA pairs.
  This asymmetry is by design: DRS is more expensive, and its K-consecutive check
  is correspondingly more stringent in epoch-time.

Usage:
    python metrics/stabilization.py --config configs/resnet18_cifar10.yaml

Output:
    results/stabilization_cka.csv      — per-pair CKA annotated with consecutive count
    results/stabilization_drs.csv      — per-pair DRS annotated with consecutive count
    results/stabilization_summary.csv  — detected t* values and recommendation
    results/stabilization.log
"""

import argparse
import csv
import logging
import os
import sys

import yaml


# ---------------------------------------------------------------------------
# Setup utilities
# ---------------------------------------------------------------------------

def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("stabilization")
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
# CSV loading
# ---------------------------------------------------------------------------

def load_cka_results(cka_csv_path: str) -> list[dict]:
    """
    Load cka_results.csv produced by metrics/cka.py.

    Returns a list of dicts sorted by epoch_curr ascending.
    Each dict has keys: epoch_prev, epoch_curr, cka_value, cka_change, below_tau.
    """
    if not os.path.exists(cka_csv_path):
        raise FileNotFoundError(
            f"CKA results not found: {cka_csv_path}\n"
            "Run: python metrics/cka.py --config configs/resnet18_cifar10.yaml"
        )

    rows = []
    with open(cka_csv_path, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append({
                "epoch_prev": int(row["epoch_prev"]),
                "epoch_curr": int(row["epoch_curr"]),
                "cka_value": float(row["cka_value"]),
                "cka_change": float(row["cka_change"]),
                "below_tau": int(row["below_tau"]),
            })

    rows.sort(key=lambda r: r["epoch_curr"])
    return rows


def load_drs_results(drs_csv_path: str) -> list[dict] | None:
    """
    Load drs_results.csv produced by metrics/drs.py.
    Returns None (with no error) if the file does not exist yet — DRS is optional.

    Each dict has keys: epoch_prev, epoch_curr, drs_value, drs_change, below_tau.
    """
    if not os.path.exists(drs_csv_path):
        return None

    rows = []
    with open(drs_csv_path, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append({
                "epoch_prev": int(row["epoch_prev"]),
                "epoch_curr": int(row["epoch_curr"]),
                "drs_value": float(row["drs_value"]),
                "drs_change": float(row["drs_change"]),
                "below_tau": int(row["below_tau"]),
            })

    rows.sort(key=lambda r: r["epoch_curr"])
    return rows


# ---------------------------------------------------------------------------
# Core stabilization detection
# ---------------------------------------------------------------------------

def detect_stabilization(
    metric_pairs: list[dict],
    tau: float,
    K: int,
    change_key: str,
) -> tuple[int | None, list[dict]]:
    """
    Apply the (τ, K) stabilization criterion to a sequence of metric pairs.

    Scans pairs in epoch order. Maintains a running consecutive count of how many
    pairs in a row have below_tau == 1. When the count first reaches exactly K,
    the current pair's epoch_curr is recorded as t*. Pairs after that continue
    to be annotated (to show whether the run holds or breaks) but do not change t*.

    If a pair has below_tau == 0, the consecutive count resets to zero — patience
    is lost and the count must climb back to K from scratch.

    Args:
        metric_pairs: sorted list of pair dicts, each containing 'epoch_curr',
                      'below_tau', and the named change_key
        tau:          the threshold — change < tau means below_tau (already in the rows;
                      tau is passed here only for documentation in the output
        K:            number of consecutive below-tau pairs required to declare t*
        change_key:   "cka_change" or "drs_change" — used to populate output rows

    Returns:
        t_star:          the epoch_curr where K consecutive below-tau pairs first
                         completed, or None if K was never reached in the data
        annotated_pairs: the input pairs with two new fields added:
                           consecutive_count — running count at this pair
                           is_t_star_trigger — 1 only on the pair that first reached K
    """
    consecutive_count = 0
    t_star = None
    annotated_pairs = []

    for pair in metric_pairs:
        is_below_tau = bool(pair["below_tau"])

        if is_below_tau:
            consecutive_count += 1
        else:
            consecutive_count = 0

        # The trigger fires exactly once: the first time consecutive_count reaches K.
        # After t_star is set, consecutive_count may continue to grow (run holds) or
        # reset (run breaks after t*), but neither changes the declared t*.
        is_trigger = (consecutive_count == K) and (t_star is None)
        if is_trigger:
            t_star = pair["epoch_curr"]

        annotated_pair = dict(pair)
        annotated_pair["consecutive_count"] = consecutive_count
        annotated_pair["is_t_star_trigger"] = int(is_trigger)
        annotated_pairs.append(annotated_pair)

    return t_star, annotated_pairs


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_pairs_table(
    pairs: list[dict],
    change_key: str,
    metric_name: str,
    tau: float,
    K: int,
    t_star: int | None,
    logger: logging.Logger,
) -> None:
    """Print a human-readable table of all pairs and their stabilization state."""
    logger.info(f"  {metric_name} pairs (τ={tau}, K={K}, detected t*={t_star}):")
    logger.info(
        f"  {'epoch_prev':>10}  {'epoch_curr':>10}  "
        f"{'change':>8}  {'below_τ':>7}  {'consec':>6}  {'trigger':>7}"
    )
    logger.info("  " + "-" * 60)

    for pair in pairs:
        trigger_marker = " ← t*" if pair["is_t_star_trigger"] else ""
        below_marker = "YES" if pair["below_tau"] else "no "
        logger.info(
            f"  {pair['epoch_prev']:>10}  {pair['epoch_curr']:>10}  "
            f"  {pair[change_key]:>6.4f}  {below_marker:>7}  "
            f"{pair['consecutive_count']:>6}  "
            f"{pair['is_t_star_trigger']:>7}{trigger_marker}"
        )


# ---------------------------------------------------------------------------
# CSV saving
# ---------------------------------------------------------------------------

def save_annotated_pairs_csv(
    annotated_pairs: list[dict],
    change_key: str,
    value_key: str,
    output_path: str,
    logger: logging.Logger,
) -> None:
    """
    Save the annotated pairs (with consecutive_count and is_t_star_trigger) to CSV.
    """
    fieldnames = [
        "epoch_prev",
        "epoch_curr",
        value_key,
        change_key,
        "below_tau",
        "consecutive_count",
        "is_t_star_trigger",
    ]

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for pair in annotated_pairs:
            writer.writerow({k: pair[k] for k in fieldnames})

    logger.info(f"  Saved annotated pairs to: {output_path}")


def save_summary_csv(
    tau: float,
    K: int,
    t_star_cka: int | None,
    t_star_drs: int | None,
    t_star_joint: int | None,
    drs_available: bool,
    n_cka_pairs: int,
    n_drs_pairs: int,
    output_path: str,
    logger: logging.Logger,
) -> None:
    """Save the summary of detected t* values to CSV."""
    rows = [
        {
            "metric": "CKA",
            "t_star": t_star_cka if t_star_cka is not None else "NOT_DETECTED",
            "tau": tau,
            "K": K,
            "n_pairs": n_cka_pairs,
            "note": "t* = first epoch_curr completing K consecutive below-tau pairs",
        },
        {
            "metric": "DRS",
            "t_star": t_star_drs if t_star_drs is not None else (
                "NOT_AVAILABLE" if not drs_available else "NOT_DETECTED"
            ),
            "tau": tau,
            "K": K,
            "n_pairs": n_drs_pairs,
            "note": (
                "DRS not computed yet" if not drs_available
                else "t* = first epoch_curr completing K consecutive below-tau pairs"
            ),
        },
        {
            "metric": "JOINT",
            "t_star": t_star_joint if t_star_joint is not None else "NOT_DETECTED",
            "tau": tau,
            "K": K,
            "n_pairs": n_cka_pairs + n_drs_pairs,
            "note": "max(t*_CKA, t*_DRS) — both metrics must satisfy the criterion",
        },
    ]

    fieldnames = ["metric", "t_star", "tau", "K", "n_pairs", "note"]
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"  Saved summary to: {output_path}")


# ---------------------------------------------------------------------------
# Joint t* computation and recommendation
# ---------------------------------------------------------------------------

def compute_joint_t_star(
    t_star_cka: int | None,
    t_star_drs: int | None,
    drs_available: bool,
    logger: logging.Logger,
) -> int | None:
    """
    Compute the joint t* from individual CKA and DRS detections.

    When DRS results exist: joint = max(t*_CKA, t*_DRS), requiring BOTH.
    When DRS is absent: joint = t*_CKA with a warning (CKA-only is weaker evidence).
    If either required metric has no detection: joint = None.
    """
    if not drs_available:
        logger.warning(
            "DRS results not found. Falling back to CKA-only detection. "
            "This is weaker evidence: CKA alone can be high even when functional "
            "behavior differs (Davari et al. 2022). Run metrics/drs.py and re-run "
            "this script for the joint criterion."
        )
        return t_star_cka

    if t_star_cka is None and t_star_drs is None:
        return None

    if t_star_cka is None:
        logger.warning(
            f"CKA stabilization not detected but DRS detected t*={t_star_drs}. "
            "Joint t* cannot be declared — CKA is still changing."
        )
        return None

    if t_star_drs is None:
        logger.warning(
            f"DRS stabilization not detected but CKA detected t*={t_star_cka}. "
            "Joint t* cannot be declared — DRS is still changing. "
            "This may mean training has not run long enough for K DRS pairs to accumulate."
        )
        return None

    joint = max(t_star_cka, t_star_drs)
    return joint


def log_recommendation(
    t_star_joint: int | None,
    t_star_cka: int | None,
    drs_available: bool,
    logger: logging.Logger,
) -> None:
    """Print a clear, actionable recommendation for the next pipeline step."""
    logger.info("=" * 70)
    logger.info("RECOMMENDATION")
    logger.info("=" * 70)

    if t_star_joint is not None:
        logger.info(f"Stabilization detected at epoch t* = {t_star_joint}")
        logger.info("")
        logger.info("Next steps:")
        logger.info(f"  1. python extract.py --epoch {t_star_joint}")
        logger.info(f"  2. python surrogates/linear_probe.py   --epoch {t_star_joint}")
        logger.info(f"  3. python surrogates/lightgbm_probe.py --epoch {t_star_joint}")
        logger.info(f"  4. python surrogates/rf_probe.py       --epoch {t_star_joint}")
        logger.info("")
        logger.info(
            "  To compare against additional freeze points, also run extract.py "
            f"at epochs {t_star_joint // 2} (early) and the final epoch (full convergence)."
        )
    elif t_star_cka is not None and not drs_available:
        logger.info(f"CKA-only stabilization detected at epoch t* = {t_star_cka}")
        logger.info("  (DRS not available — this is weaker evidence)")
        logger.info("")
        logger.info("Next steps:")
        logger.info(f"  1. python extract.py --epoch {t_star_cka}")
        logger.info(f"  2. Run surrogates with --epoch {t_star_cka}")
        logger.info("")
        logger.info(
            "  To get the joint criterion, run metrics/drs.py then re-run this script."
        )
    else:
        logger.info("Stabilization NOT detected within the available training data.")
        logger.info("")
        if not drs_available:
            logger.info("  Possible causes:")
            logger.info("    - Training may not have run long enough (increase epochs)")
            logger.info(
                f"    - τ may be too tight — try increasing tau in the config "
                "(current value is shown above)"
            )
            logger.info("    - DRS not available — run metrics/drs.py for joint criterion")
        else:
            logger.info("  Possible causes:")
            logger.info("    - Training may not have run long enough (increase epochs)")
            logger.info(
                "    - τ or K may need adjustment — check the annotated pair CSVs "
                "to see how close the metrics came to satisfying the criterion"
            )
            logger.info(
                "    - The representation may genuinely not stabilize "
                "(this is also a finding)"
            )

    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Parse arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Detect t* from CKA and DRS metric curves using the (τ, K) criterion. "
            "Run metrics/cka.py (and optionally metrics/drs.py) first."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/resnet18_cifar10.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--cka-csv",
        type=str,
        default=None,
        help="Override path to cka_results.csv (default: from config paths.results).",
    )
    parser.add_argument(
        "--drs-csv",
        type=str,
        default=None,
        help="Override path to drs_results.csv (default: from config paths.results).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override the seed from config. Input/output paths are routed to "
            "seed-specific subdirectories (e.g. results/seed_42/)."
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 2. Load config
    # ------------------------------------------------------------------
    config = load_config(args.config)

    # --seed routes all paths to a seed-specific subdirectory so
    # concurrent seed runs don't collide.
    if args.seed is not None:
        config["seed"] = args.seed
        config["paths"]["checkpoints"] = os.path.join(
            config["paths"]["checkpoints"], f"seed_{args.seed}"
        )
        config["paths"]["activations"] = os.path.join(
            config["paths"]["activations"], f"seed_{args.seed}"
        )
        config["paths"]["results"] = os.path.join(
            config["paths"]["results"], f"seed_{args.seed}"
        )

    results_dir = config["paths"]["results"]
    tau = config["tau"]
    K = config["K"]

    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Logging
    # ------------------------------------------------------------------
    log_path = os.path.join(results_dir, "stabilization.log")
    logger = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("Stabilization Detection — (τ, K) Criterion")
    logger.info("=" * 70)
    logger.info(f"Config:  {args.config}")
    logger.info(f"τ (tau): {tau}   (change threshold; change = 1 − metric_value)")
    logger.info(f"K:       {K}    (consecutive below-τ pairs required to declare t*)")

    # ------------------------------------------------------------------
    # 4. Resolve CSV paths
    # ------------------------------------------------------------------
    cka_csv_path = args.cka_csv or os.path.join(results_dir, "cka_results.csv")
    drs_csv_path = args.drs_csv or os.path.join(results_dir, "drs_results.csv")

    logger.info(f"CKA CSV: {cka_csv_path}")
    logger.info(f"DRS CSV: {drs_csv_path}")

    # ------------------------------------------------------------------
    # 5. Load CKA results (required)
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("Loading CKA results...")
    cka_pairs = load_cka_results(cka_csv_path)
    logger.info(f"  Loaded {len(cka_pairs)} CKA pairs.")

    if len(cka_pairs) == 0:
        logger.error("CKA results file is empty. Run metrics/cka.py first.")
        sys.exit(1)

    n_cka_below_tau = sum(p["below_tau"] for p in cka_pairs)
    logger.info(
        f"  Pairs below τ={tau}: {n_cka_below_tau} / {len(cka_pairs)} "
        f"({100 * n_cka_below_tau / len(cka_pairs):.1f}%)"
    )

    # ------------------------------------------------------------------
    # 6. Load DRS results (optional)
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("Loading DRS results (optional)...")
    drs_pairs = load_drs_results(drs_csv_path)
    drs_available = drs_pairs is not None

    if drs_available:
        logger.info(f"  Loaded {len(drs_pairs)} DRS pairs.")
        n_drs_below_tau = sum(p["below_tau"] for p in drs_pairs)
        logger.info(
            f"  Pairs below τ={tau}: {n_drs_below_tau} / {len(drs_pairs)} "
            f"({100 * n_drs_below_tau / len(drs_pairs):.1f}%)"
        )
    else:
        logger.info("  DRS results not found — will use CKA-only detection.")

    # ------------------------------------------------------------------
    # 7. Detect t* for CKA
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info(f"Running stabilization detection on CKA (τ={tau}, K={K})...")

    t_star_cka, cka_annotated = detect_stabilization(
        metric_pairs=cka_pairs,
        tau=tau,
        K=K,
        change_key="cka_change",
    )

    if t_star_cka is not None:
        logger.info(f"  CKA stabilization detected at epoch t* = {t_star_cka}")
    else:
        logger.info(
            f"  CKA stabilization NOT detected "
            f"(K={K} consecutive below-τ pairs never accumulated)"
        )

    log_pairs_table(
        pairs=cka_annotated,
        change_key="cka_change",
        metric_name="CKA",
        tau=tau,
        K=K,
        t_star=t_star_cka,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # 8. Detect t* for DRS (if available)
    # ------------------------------------------------------------------
    t_star_drs = None
    drs_annotated = []

    if drs_available:
        logger.info("-" * 70)
        logger.info(f"Running stabilization detection on DRS (τ={tau}, K={K})...")

        t_star_drs, drs_annotated = detect_stabilization(
            metric_pairs=drs_pairs,
            tau=tau,
            K=K,
            change_key="drs_change",
        )

        if t_star_drs is not None:
            logger.info(f"  DRS stabilization detected at epoch t* = {t_star_drs}")
        else:
            logger.info(
                f"  DRS stabilization NOT detected "
                f"(K={K} consecutive below-τ pairs never accumulated)"
            )

        log_pairs_table(
            pairs=drs_annotated,
            change_key="drs_change",
            metric_name="DRS",
            tau=tau,
            K=K,
            t_star=t_star_drs,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # 9. Compute joint t*
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("Computing joint t*...")

    t_star_joint = compute_joint_t_star(
        t_star_cka=t_star_cka,
        t_star_drs=t_star_drs,
        drs_available=drs_available,
        logger=logger,
    )

    logger.info(f"  t*_CKA   = {t_star_cka}")
    logger.info(f"  t*_DRS   = {t_star_drs if drs_available else 'N/A (not computed)'}")
    logger.info(f"  t*_JOINT = {t_star_joint}")

    # ------------------------------------------------------------------
    # 10. Save annotated CSVs
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("Saving annotated results...")

    cka_output_path = os.path.join(results_dir, "stabilization_cka.csv")
    save_annotated_pairs_csv(
        annotated_pairs=cka_annotated,
        change_key="cka_change",
        value_key="cka_value",
        output_path=cka_output_path,
        logger=logger,
    )

    if drs_available and drs_annotated:
        drs_output_path = os.path.join(results_dir, "stabilization_drs.csv")
        save_annotated_pairs_csv(
            annotated_pairs=drs_annotated,
            change_key="drs_change",
            value_key="drs_value",
            output_path=drs_output_path,
            logger=logger,
        )

    summary_output_path = os.path.join(results_dir, "stabilization_summary.csv")
    save_summary_csv(
        tau=tau,
        K=K,
        t_star_cka=t_star_cka,
        t_star_drs=t_star_drs,
        t_star_joint=t_star_joint,
        drs_available=drs_available,
        n_cka_pairs=len(cka_pairs),
        n_drs_pairs=len(drs_pairs) if drs_available else 0,
        output_path=summary_output_path,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # 11. Print recommendation
    # ------------------------------------------------------------------
    log_recommendation(
        t_star_joint=t_star_joint,
        t_star_cka=t_star_cka,
        drs_available=drs_available,
        logger=logger,
    )


if __name__ == "__main__":
    main()
