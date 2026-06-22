"""
Generate the core CIFAR-10 figures from the ext_*.csv outputs.

Reads study_trajectories.csv (from the aggregator) and the per-run ext_*.csv, writes PNGs.
Headless-safe (Agg backend). Usage from repo root:
    python -m analysis.plots_extended \
        --base /local/data/gme101 \
        --summary /local/data/gme101/_study_summary \
        --out /local/data/gme101/_figs --lock-k 50
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_concat(base, runs, name):
    fr = []
    for r in runs:
        p = os.path.join(base, r, "results", f"{name}.csv")
        if os.path.exists(p):
            fr.append(pd.read_csv(p))
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()


def fig_timing(traj, subs, out, lock_k):
    """Per-LR: probe accuracy vs CKA-to-final vs SVCCA-to-final vs top-k subspace lock."""
    lrs = sorted(traj.lr.unique())
    fig, axes = plt.subplots(1, len(lrs), figsize=(5.2 * len(lrs), 4.2), sharey=True, squeeze=False)
    pa = (subs[subs.k == lock_k].groupby(["lr", "epoch"]).pa_mean_cos.mean().reset_index()
          if not subs.empty and "pa_mean_cos" in subs.columns else pd.DataFrame())
    for ax, lr in zip(axes[0], lrs):
        t = traj[traj.lr == lr].sort_values("epoch")
        if "test_acc_mean" in t:
            ax.plot(t.epoch, t.test_acc_mean, color="black", lw=2, label="probe acc")
        if "cka_debiased_tofinal_mean" in t:
            ax.plot(t.epoch, t.cka_debiased_tofinal_mean, color="C3", lw=2, label="CKA-to-final (debiased)")
        if "svcca_tofinal_mean" in t:
            ax.plot(t.epoch, t.svcca_tofinal_mean, color="C0", lw=2, label="SVCCA-to-final (99% var)")
        if not pa.empty:
            pl = pa[pa.lr == lr]
            ax.plot(pl.epoch, pl.pa_mean_cos, color="C2", lw=2, ls="--",
                    label=f"top-{lock_k} subspace lock (cos)")
        ax.set_title(f"\u03b7 = {lr}")
        ax.set_xlabel("epoch"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("accuracy / similarity")
    axes[0][-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Functional maturation precedes geometric stabilization", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_timing.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_rank_heatmap(rank_df, out, rep_run):
    d = rank_df[(rank_df.run == rep_run) & (rank_df.basis == "self")]
    if d.empty:
        return
    piv = d.pivot_table(index="k", columns="epoch", values="test_acc")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="viridis",
                   extent=[piv.columns.min(), piv.columns.max(), 0, len(piv.index)])
    ax.set_yticks(np.arange(len(piv.index)) + 0.5); ax.set_yticklabels(piv.index)
    ax.set_xlabel("epoch"); ax.set_ylabel("retained rank k")
    ax.set_title(f"Probe accuracy vs retained rank over training ({rep_run})")
    fig.colorbar(im, ax=ax, label="test acc")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_rank_accuracy.png"), dpi=150)
    plt.close(fig)


def fig_spectral_mass(subs, out, rep_run):
    d = subs[subs.run == rep_run]
    if d.empty:
        return
    ep = d.epoch.max()
    d = d[d.epoch == ep].sort_values("k")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d.k, d.sv_mass, "o-", label="singular-value mass  \u03a3\u03c3/\u03a3\u03c3")
    ax.plot(d.k, d.energy_mass, "s-", label="energy mass  \u03a3\u03c3\u00b2/\u03a3\u03c3\u00b2 (= explained var)")
    ax.axhline(0.8, color="grey", ls=":", lw=1)
    ax.set_xlabel("retained rank k"); ax.set_ylabel("captured mass")
    ax.set_title(f"Why the 80% split is near-tautological (epoch {ep}, {rep_run})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_spectral_mass.png"), dpi=150)
    plt.close(fig)


def fig_component_cka(subs, out, lock_k):
    if subs.empty or "cka_main_tofinal" not in subs.columns:
        return
    d = subs[subs.k == lock_k].groupby("epoch")[["cka_main_tofinal", "cka_resid_tofinal"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d.epoch, d.cka_main_tofinal, "o-", color="C3", label=f"main (top-{lock_k})")
    ax.plot(d.epoch, d.cka_resid_tofinal, "s-", color="C0", label="residual")
    ax.set_xlabel("epoch"); ax.set_ylabel("CKA to final"); ax.set_ylim(0, 1.02)
    ax.set_title("Dominant subspace aligns early; residual reorganizes late")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_component_cka.png"), dpi=150)
    plt.close(fig)


def fig_fisher(fish, out):
    if fish.empty:
        return
    ep = fish.epoch.max()
    d = fish[fish.epoch == ep].groupby("component").fisher.mean().reset_index().sort_values("component")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d.component, d.fisher, lw=1.5)
    ax.set_xlabel("singular component index"); ax.set_ylabel("Fisher ratio (between/within)")
    ax.set_title(f"Class-discriminability by component (epoch {ep}, mean over runs)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_fisher.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/local/data/gme101")
    ap.add_argument("--summary", default="/local/data/gme101/_study_summary")
    ap.add_argument("--out", default="/local/data/gme101/_figs")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--rep-run", default="final_lr010_seed0")
    ap.add_argument("--lock-k", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.runs is None:
        runs = [os.path.basename(r) for r in sorted(glob.glob(os.path.join(args.base, "final_lr*_seed*")))
                if os.path.isdir(r)]
    else:
        runs = args.runs

    traj = pd.read_csv(os.path.join(args.summary, "study_trajectories.csv"))
    subs = load_concat(args.base, runs, "ext_subspace")
    rank = load_concat(args.base, runs, "ext_rank_accuracy")
    fish = load_concat(args.base, runs, "ext_fisher")

    rep = args.rep_run if args.rep_run in runs else (runs[0] if runs else None)
    fig_timing(traj, subs, args.out, args.lock_k)
    if rep:
        fig_rank_heatmap(rank, args.out, rep)
        fig_spectral_mass(subs, args.out, rep)
    fig_component_cka(subs, args.out, args.lock_k)
    fig_fisher(fish, args.out)
    print("wrote figures to", args.out, ":", sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
