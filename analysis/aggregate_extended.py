"""
Aggregate the per-run ext_*.csv files across the 15 runs into study-level summaries and,
crucially, a PAIRED statistical test of the headline timing gap.

Headline claim made defensible: for each seed compute
    t_suff  = first epoch the full-feature probe reaches (final_probe_acc - delta_pp)
    t_stab  = first epoch cka_debiased_tofinal >= cka_thresh
    t_svcca = first epoch svcca_tofinal      >= svcca_thresh
then test the per-seed differences (t_stab - t_suff) and (t_svcca - t_suff) with a
paired Wilcoxon signed-rank test and a bootstrap CI of the mean gap. The SVCCA gap is the
"matched-invariance" version: it should be much smaller than the CKA gap if the CKA lag is
largely the GL-vs-orthogonal invariance discrepancy.

Outputs (written to --out):
  study_trajectories.csv   epoch x metric, mean & sd across seeds within each lr
  study_milestones.csv     one row per run: t_suff, t_stab, t_svcca, gaps
  study_timing_tests.csv   per lr and pooled: mean gap, bootstrap CI, Wilcoxon p
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def first_crossing(epochs, values, thresh, direction=">="):
    e = np.asarray(epochs, float)
    v = np.asarray(values, float)
    order = np.argsort(e)
    e, v = e[order], v[order]
    mask = (v >= thresh) if direction == ">=" else (v <= thresh)
    idx = np.where(mask)[0]
    return float(e[idx[0]]) if len(idx) else np.nan


def boot_ci(x, n=10000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return (float(x.mean()), float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def load_concat(base, runs, name):
    frames = []
    for r in runs:
        p = os.path.join(base, r, "results", f"{name}.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/local/data/gme101")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out", default="/local/data/gme101/_study_summary")
    ap.add_argument("--delta-pp", type=float, default=2.0, help="probe sufficiency band (pp below final probe acc)")
    ap.add_argument("--cka-thresh", type=float, default=0.95)
    ap.add_argument("--svcca-thresh", type=float, default=0.95)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.runs is None:
        runs = [os.path.basename(r) for r in sorted(glob.glob(os.path.join(args.base, "final_lr*_seed*")))
                if os.path.isdir(r)]
    else:
        runs = args.runs

    sim = load_concat(args.base, runs, "ext_similarity")
    fun = load_concat(args.base, runs, "ext_probe_functional")
    if sim.empty or fun.empty:
        raise SystemExit("missing ext_similarity / ext_probe_functional; run the driver first")

    # ---- trajectories: mean & sd across seeds within lr ----
    sim_metrics = [c for c in ["cka_biased_tofinal", "cka_debiased_tofinal", "svcca_tofinal",
                               "cka_biased_consec", "cka_debiased_consec"] if c in sim.columns]
    traj = (sim.groupby(["lr", "epoch"])[sim_metrics].agg(["mean", "std"]).reset_index())
    traj.columns = ["_".join([c for c in col if c]).strip("_") for col in traj.columns.values]
    fun_metrics = [c for c in ["test_acc", "test_nll", "test_margin"] if c in fun.columns]
    funtraj = (fun.groupby(["lr", "epoch"])[fun_metrics].agg(["mean", "std"]).reset_index())
    funtraj.columns = ["_".join([c for c in col if c]).strip("_") for col in funtraj.columns.values]
    traj.merge(funtraj, on=["lr", "epoch"], how="outer").sort_values(["lr", "epoch"]).to_csv(
        os.path.join(args.out, "study_trajectories.csv"), index=False)

    # ---- per-run milestones ----
    rows = []
    for run in runs:
        s = sim[sim.run == run].sort_values("epoch")
        f = fun[fun.run == run].sort_values("epoch")
        if s.empty or f.empty:
            continue
        final_acc = f.test_acc.iloc[-1]
        t_suff = first_crossing(f.epoch, f.test_acc, final_acc - args.delta_pp / 100.0, ">=")
        t_stab = first_crossing(s.epoch, s.cka_debiased_tofinal, args.cka_thresh, ">=") \
            if "cka_debiased_tofinal" in s else np.nan
        t_svcca = first_crossing(s.epoch, s.svcca_tofinal, args.svcca_thresh, ">=") \
            if "svcca_tofinal" in s else np.nan
        lr = s.lr.iloc[0]; seed = s.seed.iloc[0]
        rows.append(dict(run=run, lr=lr, seed=seed, final_probe_acc=final_acc,
                         t_suff=t_suff, t_stab_cka=t_stab, t_stab_svcca=t_svcca,
                         gap_cka=t_stab - t_suff, gap_svcca=t_svcca - t_suff))
    mil = pd.DataFrame(rows)
    mil.to_csv(os.path.join(args.out, "study_milestones.csv"), index=False)

    # ---- paired tests per lr and pooled ----
    test_rows = []
    def run_test(sub, label):
        for gapcol in ["gap_cka", "gap_svcca"]:
            g = sub[gapcol].dropna().values
            if len(g) < 2:
                continue
            mean, lo, hi = boot_ci(g)
            try:
                w = wilcoxon(g)
                p = float(w.pvalue)
            except ValueError:
                p = np.nan  # all-zero differences etc.
            test_rows.append(dict(group=label, gap=gapcol, n=len(g),
                                  mean_gap=mean, ci_lo=lo, ci_hi=hi, wilcoxon_p=p))
    for lr, sub in mil.groupby("lr"):
        run_test(sub, f"lr={lr}")
    run_test(mil, "pooled")
    pd.DataFrame(test_rows).to_csv(os.path.join(args.out, "study_timing_tests.csv"), index=False)

    print("wrote:", os.listdir(args.out))
    if test_rows:
        print(pd.DataFrame(test_rows).to_string(index=False))


if __name__ == "__main__":
    main()
