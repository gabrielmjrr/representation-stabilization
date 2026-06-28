"""
scripts/verify_dense_analysis.py

After re-running analysis/run_extended_analysis.py + aggregate_extended.py
on the densified runs, verify that:

  1. All original-epoch rows in the new ext_*.csv match the old values within
     tolerance (confirms re-analysis is deterministic and nothing was corrupted).
  2. The new fine-grained first-crossing epochs in the dense window are reported
     (CKA stabilization, SVCCA stabilization, probe sufficiency).

The "original" epochs are the 38 checkpoint epochs from the committed config:
  0-20 (dense), 22,24,26,28,30, 35,40,45,50, 60-200 (step 10).

Usage
-----
# old results = committed thesis_results; new results = freshly synced cluster output
python scripts/verify_dense_analysis.py \
    --old-base ~/thesis_results \
    --new-base ~/thesis_results          # same dir if files were overwritten in-place

# Or point new-base at a separate directory where you synced the post-densification CSVs:
python scripts/verify_dense_analysis.py \
    --old-base ~/thesis_results_pre_dense \
    --new-base ~/thesis_results

Notes
-----
* If old-base and new-base are the same directory, the old values must have been
  committed to git before re-analysis. Retrieve them with:
      git show HEAD:thesis_results/<run>/results/ext_similarity.csv > /tmp/old_sim.csv
  and point --old-base at a directory pre-populated from git show.
* Probe metrics (test_acc, test_nll, test_margin) use rtol=1e-3 because logistic
  regression convergence can vary by a few ULP across platforms. CKA/SVCCA columns
  use rtol=1e-5 (purely deterministic numpy computation).
* The crossing-gap check compares old vs new study_milestones.csv (from
  aggregate_extended.py). The crossing EPOCH may change on the finer grid (that's
  the scientific result), but the crossing VALUES should remain consistent.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Original checkpoint epochs (from committed config checkpoint_epochs list)
# ---------------------------------------------------------------------------
ORIGINAL_EPOCHS = set([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    22, 24, 26, 28, 30, 35, 40, 45, 50,
    60, 70, 80, 90, 100, 110, 120, 130, 140,
    150, 160, 170, 180, 190, 200,
])

# Thesis-critical columns — flag violations in these explicitly
_SIM_THESIS_COLS   = ["cka_biased_tofinal", "cka_debiased_tofinal", "svcca_tofinal"]
_PROBE_THESIS_COLS = ["test_acc", "test_nll", "test_margin"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _load(path: str) -> pd.DataFrame | None:
    return pd.read_csv(path) if os.path.exists(path) else None


def _check_overlap(old_df: pd.DataFrame, new_df: pd.DataFrame,
                   csv_name: str, rtol: float) -> list[str]:
    """
    For each original epoch present in old_df, check that new_df has the same
    numeric values within rtol. Returns a list of human-readable violation strings.
    """
    violations = []
    if "epoch" not in old_df.columns:
        return violations

    old_orig = old_df[old_df.epoch.isin(ORIGINAL_EPOCHS)].sort_values("epoch")
    new_idx  = {int(r["epoch"]): r for _, r in new_df.iterrows()} if new_df is not None else {}

    numeric_cols = [
        c for c in old_orig.columns
        if pd.api.types.is_numeric_dtype(old_orig[c]) and c not in ("epoch",)
    ]

    for _, old_row in old_orig.iterrows():
        ep = int(old_row["epoch"])
        if ep not in new_idx:
            violations.append(f"  {csv_name} epoch {ep:3d}: MISSING in new results")
            continue
        new_row = new_idx[ep]
        for col in numeric_cols:
            if col not in new_row.index:
                continue
            ov = float(old_row[col])
            nv = float(new_row[col])
            if np.isnan(ov) and np.isnan(nv):
                continue
            if np.isnan(ov) or np.isnan(nv):
                # one is NaN and the other isn't — always a violation
                violations.append(
                    f"  {csv_name} epoch {ep:3d} [{col}]: old={ov} new={nv} (NaN mismatch)"
                )
                continue
            rel = abs(ov - nv) / (abs(ov) + 1e-12)
            if rel > rtol:
                marker = " *** THESIS" if col in _SIM_THESIS_COLS + _PROBE_THESIS_COLS else ""
                violations.append(
                    f"  {csv_name} epoch {ep:3d} [{col}]: old={ov:.6f} new={nv:.6f} "
                    f"reldiff={rel:.2e} > rtol={rtol}{marker}"
                )
    return violations


def _report_crossings(new_sim: pd.DataFrame | None,
                      new_probe: pd.DataFrame | None,
                      cka_thresh: float,
                      svcca_thresh: float,
                      delta_pp: float) -> list[str]:
    """
    Report first-crossing epochs for the thesis-critical milestones on the new
    (denser) epoch grid.
    """
    lines = []

    if new_sim is not None:
        s = new_sim.sort_values("epoch")

        for col, thresh, label in [
            ("cka_debiased_tofinal", cka_thresh,   f"t_stab_cka  (cka_debiased>={cka_thresh})"),
            ("svcca_tofinal",        svcca_thresh,  f"t_stab_svcca(svcca>={svcca_thresh})"),
        ]:
            if col not in s.columns:
                continue
            idx = np.where(s[col].values >= thresh)[0]
            t = int(s.epoch.iloc[idx[0]]) if len(idx) else None
            lines.append(f"  {label}: epoch {t if t is not None else 'NOT CROSSED'}")

    if new_probe is not None and "test_acc" in new_probe.columns:
        p = new_probe.sort_values("epoch")
        final_acc = float(p.test_acc.iloc[-1])
        thresh = final_acc - delta_pp / 100.0
        idx = np.where(p.test_acc.values >= thresh)[0]
        t = int(p.epoch.iloc[idx[0]]) if len(idx) else None
        lines.append(
            f"  t_suff       (probe>={thresh:.4f} = final-{delta_pp}%): "
            f"epoch {t if t is not None else 'NOT CROSSED'}"
        )

    return lines


def _milestone_gap_summary(old_mil: pd.DataFrame | None,
                            new_mil: pd.DataFrame | None) -> list[str]:
    """
    Compare gap_cka and gap_svcca between old and new study_milestones.csv.
    The gap value (mean/CI) may shift because crossings are now resolved at finer
    granularity — report both and the delta.
    """
    lines = []
    if old_mil is None or new_mil is None:
        return lines

    for gap_col in ["gap_cka", "gap_svcca"]:
        if gap_col not in old_mil.columns or gap_col not in new_mil.columns:
            continue
        old_vals = old_mil[gap_col].dropna()
        new_vals = new_mil[gap_col].dropna()
        if old_vals.empty or new_vals.empty:
            continue
        lines.append(
            f"  {gap_col}:  old mean={old_vals.mean():.2f} (n={len(old_vals)})  "
            f"new mean={new_vals.mean():.2f} (n={len(new_vals)})  "
            f"delta={new_vals.mean()-old_vals.mean():+.2f}"
        )
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify new dense ext_*.csv reproduces old original-epoch values "
            "and report fine-grained crossing epochs."
        )
    )
    parser.add_argument(
        "--old-base", required=True,
        help=(
            "Root directory containing old (pre-densification) per-run results. "
            "E.g. ~/thesis_results.  Each run lives at <old-base>/<run>/results/."
        ),
    )
    parser.add_argument(
        "--new-base", required=True,
        help=(
            "Root directory containing new (post-densification) per-run results. "
            "Can equal --old-base if files were overwritten in-place."
        ),
    )
    parser.add_argument(
        "--runs", nargs="*", default=None,
        help="Run names to check. Default: all final_lr*_seed* dirs found in --old-base.",
    )
    parser.add_argument(
        "--rtol-sim", type=float, default=1e-5,
        help="Relative tolerance for CKA/SVCCA columns (default 1e-5).",
    )
    parser.add_argument(
        "--rtol-probe", type=float, default=1e-3,
        help="Relative tolerance for probe accuracy columns (default 1e-3).",
    )
    parser.add_argument("--cka-thresh",   type=float, default=0.95)
    parser.add_argument("--svcca-thresh", type=float, default=0.95)
    parser.add_argument("--delta-pp",     type=float, default=2.0,
                        help="Probe sufficiency: within delta_pp%% of final acc (default 2.0).")
    parser.add_argument(
        "--study-summary-old", default=None,
        help="Path to old _study_summary dir (for gap comparison). Optional.",
    )
    parser.add_argument(
        "--study-summary-new", default=None,
        help="Path to new _study_summary dir (for gap comparison). Optional.",
    )
    args = parser.parse_args()

    old_base = os.path.expanduser(args.old_base)
    new_base = os.path.expanduser(args.new_base)

    if args.runs:
        runs = args.runs
    else:
        runs = sorted(
            os.path.basename(p)
            for p in glob.glob(os.path.join(old_base, "final_lr*_seed*"))
            if os.path.isdir(p)
        )

    if not runs:
        print(f"No runs found under {old_base}", file=sys.stderr)
        sys.exit(1)

    total_violations = 0
    print(f"Checking {len(runs)} runs")
    print(f"  rtol_sim={args.rtol_sim}  rtol_probe={args.rtol_probe}")
    print(f"  CKA threshold={args.cka_thresh}  SVCCA threshold={args.svcca_thresh}"
          f"  probe delta={args.delta_pp}%")
    print("=" * 70)

    for run in runs:
        old_dir = os.path.join(old_base, run, "results")
        new_dir = os.path.join(new_base, run, "results")
        print(f"\n{run}")

        old_sim   = _load(os.path.join(old_dir, "ext_similarity.csv"))
        new_sim   = _load(os.path.join(new_dir, "ext_similarity.csv"))
        old_probe = _load(os.path.join(old_dir, "ext_probe_functional.csv"))
        new_probe = _load(os.path.join(new_dir, "ext_probe_functional.csv"))

        if new_sim is None:
            print("  MISSING: ext_similarity.csv in new results — skipping.")
            total_violations += 1
            continue

        # ---- Overlap check ----
        violations = []
        if old_sim is not None:
            violations += _check_overlap(old_sim, new_sim, "ext_similarity", args.rtol_sim)
        if old_probe is not None and new_probe is not None:
            violations += _check_overlap(old_probe, new_probe, "ext_probe_functional",
                                         args.rtol_probe)

        if violations:
            for v in violations:
                print(v)
            total_violations += len(violations)
        else:
            old_epochs = set(old_sim["epoch"].unique()) if old_sim is not None else set()
            new_epochs = set(new_sim["epoch"].unique())
            new_only   = sorted(new_epochs - old_epochs)
            print(f"  OK: {len(old_epochs & new_epochs)} overlapping epochs match.")
            print(f"  New epochs added: {new_only}")

        # ---- Crossing report on new data ----
        crossing_lines = _report_crossings(
            new_sim, new_probe,
            cka_thresh=args.cka_thresh,
            svcca_thresh=args.svcca_thresh,
            delta_pp=args.delta_pp,
        )
        for line in crossing_lines:
            print(line)

    # ---- Study-level gap comparison (optional) ----
    if args.study_summary_old and args.study_summary_new:
        print("\n" + "=" * 70)
        print("Study-level gap comparison (study_milestones.csv)")
        old_mil = _load(os.path.join(os.path.expanduser(args.study_summary_old),
                                     "study_milestones.csv"))
        new_mil = _load(os.path.join(os.path.expanduser(args.study_summary_new),
                                     "study_milestones.csv"))
        gap_lines = _milestone_gap_summary(old_mil, new_mil)
        if gap_lines:
            for line in gap_lines:
                print(line)
        else:
            print("  (no study_milestones.csv found at one or both paths)")

    print("\n" + "=" * 70)
    if total_violations == 0:
        print("All checks passed.")
    else:
        print(f"FAILED: {total_violations} violation(s). See above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
