"""
Aggregate per-run ext_*.csv across runs into study-level summaries + paired timing tests.

t_suff comes from your CANONICAL probes (results/probe_results_long.csv, a chosen
probe_name) when available, falling back to the driver's ext_probe_functional probe.

Milestones per run:
    t_suff     first epoch canonical probe test_acc >= final - delta_pp
    t_stab_cka first epoch cka_debiased_tofinal >= cka_thresh         (full geometry)
    t_svcca    first epoch svcca_tofinal      >= svcca_thresh         (full GL subspace, 99% var)
    t_sublock  first epoch pa_mean_cos at k=lock_k >= pa_thresh       (top-k discriminative subspace)
Gaps: gap_cka, gap_svcca relative to t_suff (positive => sufficiency first);
      gap_lock_vs_cka = t_stab_cka - t_sublock (positive => top-k locks before full geometry).
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def first_crossing(epochs, values, thresh, direction=">="):
    e = np.asarray(epochs, float); v = np.asarray(values, float)
    o = np.argsort(e); e, v = e[o], v[o]
    mask = (v >= thresh) if direction == ">=" else (v <= thresh)
    idx = np.where(mask)[0]
    return float(e[idx[0]]) if len(idx) else np.nan


def boot_ci(x, n=10000, alpha=0.05, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return (float(x.mean()), float(np.percentile(m, 100*alpha/2)),
            float(np.percentile(m, 100*(1-alpha/2))))


def load_concat(base, runs, name):
    frames = []
    for r in runs:
        p = os.path.join(base, r, "results", f"{name}.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def canonical_probe_series(base, run, probe_csv, probe_name):
    """Return (epochs, test_acc) from a run's canonical probe table, or None."""
    p = os.path.join(base, run, "results", probe_csv)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    if "probe_name" in df.columns:
        df = df[df.probe_name == probe_name]
    if df.empty or "test_acc" not in df.columns:
        return None
    df = df.sort_values("epoch")
    return df.epoch.values, df.test_acc.values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/local/data/gme101")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out", default="/local/data/gme101/_study_summary")
    ap.add_argument("--delta-pp", type=float, default=2.0)
    ap.add_argument("--cka-thresh", type=float, default=0.95)
    ap.add_argument("--svcca-thresh", type=float, default=0.95)
    ap.add_argument("--pa-thresh", type=float, default=0.95)
    ap.add_argument("--lock-k", type=int, default=50, help="k for the top-k subspace-lock milestone")
    ap.add_argument("--probe-csv", default="probe_results_long.csv")
    ap.add_argument("--probe-name", default="logistic_regression")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.runs is None:
        runs = [os.path.basename(r) for r in sorted(glob.glob(os.path.join(args.base, "final_lr*_seed*")))
                if os.path.isdir(r)]
    else:
        runs = args.runs

    sim = load_concat(args.base, runs, "ext_similarity")
    fun = load_concat(args.base, runs, "ext_probe_functional")
    subs = load_concat(args.base, runs, "ext_subspace")
    if sim.empty:
        raise SystemExit("missing ext_similarity; run the driver first")

    # ---- trajectories (mean & sd across seeds within lr) ----
    sim_metrics = [c for c in ["cka_biased_tofinal", "cka_debiased_tofinal", "svcca_tofinal",
                               "cka_biased_consec", "cka_debiased_consec"] if c in sim.columns]
    traj = sim.groupby(["lr", "epoch"])[sim_metrics].agg(["mean", "std"]).reset_index()
    traj.columns = ["_".join([c for c in col if c]).strip("_") for col in traj.columns.values]
    if not fun.empty:
        fm = [c for c in ["test_acc", "test_nll", "test_margin"] if c in fun.columns]
        ft = fun.groupby(["lr", "epoch"])[fm].agg(["mean", "std"]).reset_index()
        ft.columns = ["_".join([c for c in col if c]).strip("_") for col in ft.columns.values]
        traj = traj.merge(ft, on=["lr", "epoch"], how="outer")
    traj.sort_values(["lr", "epoch"]).to_csv(os.path.join(args.out, "study_trajectories.csv"), index=False)

    # ---- per-run milestones ----
    used_canonical = 0
    rows = []
    for run in runs:
        s = sim[sim.run == run].sort_values("epoch")
        if s.empty:
            continue
        cano = canonical_probe_series(args.base, run, args.probe_csv, args.probe_name)
        if cano is not None:
            ep_a, acc = cano; used_canonical += 1; src = args.probe_name
        elif not fun.empty and (fun.run == run).any():
            f = fun[fun.run == run].sort_values("epoch"); ep_a, acc = f.epoch.values, f.test_acc.values
            src = "ext_probe_functional"
        else:
            continue
        final_acc = acc[-1]
        t_suff = first_crossing(ep_a, acc, final_acc - args.delta_pp/100.0, ">=")
        t_stab = first_crossing(s.epoch, s.cka_debiased_tofinal, args.cka_thresh) if "cka_debiased_tofinal" in s else np.nan
        t_svc = first_crossing(s.epoch, s.svcca_tofinal, args.svcca_thresh) if "svcca_tofinal" in s else np.nan
        t_lock = np.nan
        if not subs.empty:
            sl = subs[(subs.run == run) & (subs.k == args.lock_k)].sort_values("epoch")
            if not sl.empty and "pa_mean_cos" in sl:
                t_lock = first_crossing(sl.epoch, sl.pa_mean_cos, args.pa_thresh)
        rows.append(dict(run=run, lr=s.lr.iloc[0], seed=s.seed.iloc[0], probe_src=src,
                         final_acc=final_acc, t_suff=t_suff, t_stab_cka=t_stab,
                         t_svcca=t_svc, t_sublock=t_lock,
                         gap_cka=t_stab - t_suff, gap_svcca=t_svc - t_suff,
                         gap_lock_vs_cka=t_stab - t_lock))
    mil = pd.DataFrame(rows)
    mil.to_csv(os.path.join(args.out, "study_milestones.csv"), index=False)

    # ---- paired tests ----
    test_rows = []
    def run_test(sub, label):
        for col in ["gap_cka", "gap_svcca", "gap_lock_vs_cka"]:
            if col not in sub:
                continue
            g = sub[col].dropna().values
            if len(g) < 2:
                continue
            mean, lo, hi = boot_ci(g)
            try:
                p = float(wilcoxon(g).pvalue)
            except ValueError:
                p = np.nan
            test_rows.append(dict(group=label, gap=col, n=len(g), mean_gap=mean,
                                  ci_lo=lo, ci_hi=hi, wilcoxon_p=p))
    for lr, sub in mil.groupby("lr"):
        run_test(sub, f"lr={lr}")
    run_test(mil, "pooled")
    pd.DataFrame(test_rows).to_csv(os.path.join(args.out, "study_timing_tests.csv"), index=False)

    print(f"probe source: canonical={used_canonical}/{len(runs)} runs ({args.probe_name})")
    print("wrote:", sorted(os.listdir(args.out)))
    if test_rows:
        print(pd.DataFrame(test_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
