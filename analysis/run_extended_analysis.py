"""
Driver: compute the extended CIFAR-10 analyses for one or more runs and write tidy
long-format CSVs into each run's results/ directory.

Layout assumed (confirmed from the cluster dump):
  <base>/<run>/activations/full_train/features_epoch_XXXX.npy   (50000, 512)
  <base>/<run>/activations/full_train/labels.npy                (50000,)
  <base>/<run>/activations/full_test/features_epoch_XXXX.npy    (10000, 512)
  <base>/<run>/activations/full_test/labels.npy                 (10000,)
  <base>/<run>/activations/cka_subset/features_epoch_XXXX.npy   (2040, 512)  row-aligned
  <base>/<run>/activations/cka_subset/labels.npy
where <run> matches  final_lr{005,010,020}_seed{0,1,2,3,42}.

Outputs (per run, written to <run>/results/):
  ext_similarity.csv        cka biased/debiased + svcca, to-final & consecutive,
                            train-subset & held-out(test-subset)
  ext_subspace.csv          per (epoch,k): mass fractions, principal angles to final,
                            component-wise CKA(main/resid)-to-final
  ext_rank_accuracy.csv     per (epoch,k): probe acc/nll/margin in per-checkpoint basis
                            and in fixed-final basis
  ext_fisher.csv            per (epoch,component): Fisher ratio + singular value
  ext_probe_functional.csv  per epoch: full-feature probe acc/nll/margin + head-catchup

Cheap metrics (similarity, subspace, fisher, functional) run on all epochs quickly.
The rank-accuracy sweep is the expensive part (one probe fit per k per epoch); control it
with --probe-subset, --ks, and --rank-epoch-stride, and parallelise across runs.
"""
from __future__ import annotations
import argparse, glob, os, re
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # repo root
from metrics import spectral as sp
from metrics import similarity as sim
from surrogates import probes_extended as pe

LR_MAP = {"005": 0.05, "010": 0.10, "020": 0.20}
DEFAULT_KS = [1, 2, 4, 8, 16, 32, 39, 52, 64, 128, 256, 512]
DEFAULT_QS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]


# --------------------------------------------------------------------- IO utils

def parse_run(run_name: str):
    m = re.search(r"lr(\d+)_seed(\d+)", run_name)
    if not m:
        return None, None
    return LR_MAP.get(m.group(1), np.nan), int(m.group(2))


def epochs_available(run_dir: str, split: str = "cka_subset"):
    files = glob.glob(os.path.join(run_dir, "activations", split, "features_epoch_*.npy"))
    eps = sorted(int(re.search(r"epoch_(\d+)", f).group(1)) for f in files)
    return eps


def load_feat(run_dir, split, epoch):
    p = os.path.join(run_dir, "activations", split, f"features_epoch_{epoch:04d}.npy")
    return np.load(p).astype(np.float64)


def load_labels(run_dir, split):
    return np.load(os.path.join(run_dir, "activations", split, "labels.npy")).astype(int)


# ------------------------------------------------------------------- directions

def directions_via_covariance(Phi: np.ndarray, center: bool = True):
    """Fast feature-space directions V (d,d desc) and singular values s (d,) via eigh of
    the dxd covariance, avoiding an n x d SVD. s = sqrt(eigenvalues)."""
    mean = Phi.mean(0, keepdims=True) if center else np.zeros((1, Phi.shape[1]))
    Xc = Phi - mean
    C = Xc.T @ Xc
    w, V = np.linalg.eigh(C)            # ascending
    order = np.argsort(w)[::-1]
    w = np.clip(w[order], 0, None)
    V = V[:, order]
    s = np.sqrt(w)
    return V, s, mean


# --------------------------------------------------------------- per-run compute

def compute_run(run_dir: str, args) -> dict:
    lr, seed = parse_run(os.path.basename(run_dir))
    eps = epochs_available(run_dir)
    if args.max_epochs:
        eps = eps[: args.max_epochs]
    T = eps[-1]
    tag = dict(run=os.path.basename(run_dir), lr=lr, seed=seed)

    y_sub = load_labels(run_dir, "cka_subset")
    y_tr = load_labels(run_dir, "full_train")
    y_te = load_labels(run_dir, "full_test")

    # final-checkpoint references
    Phi_sub_T = load_feat(run_dir, "cka_subset", T)
    Phi_tr_T = load_feat(run_dir, "full_train", T)
    V_T, s_T, mean_T = directions_via_covariance(Phi_tr_T, center=args.center)
    # held-out CKA uses a fixed test subset of matched size n
    n_cka = Phi_sub_T.shape[0]
    rng = np.random.default_rng(0)
    te_idx = rng.choice(load_feat(run_dir, "full_test", T).shape[0], size=n_cka, replace=False)
    Phi_teSub_T = load_feat(run_dir, "full_test", T)[te_idx]

    # probe sample (subset for speed on the rank sweep)
    tr_idx = rng.choice(Phi_tr_T.shape[0], size=min(args.probe_subset, Phi_tr_T.shape[0]),
                        replace=False)

    rows_sim, rows_sub, rows_rank, rows_fish, rows_fun = [], [], [], [], []
    prev_sub = None
    Phi_teSub_prevT = None

    for ei, ep in enumerate(eps):
        Phi_sub = load_feat(run_dir, "cka_subset", ep)
        Phi_teSub = load_feat(run_dir, "full_test", ep)[te_idx]

        # ---- similarity (cheap; all epochs) ----
        row = dict(tag, epoch=ep)
        row["cka_biased_tofinal"] = sim.linear_cka(Phi_sub, Phi_sub_T)
        row["cka_debiased_tofinal"] = sim.linear_cka_debiased(Phi_sub, Phi_sub_T)
        row["svcca_tofinal"] = sim.svcca(Phi_sub, Phi_sub_T, var_keep=args.svcca_var)
        row["cka_biased_tofinal_heldout"] = sim.linear_cka(Phi_teSub, Phi_teSub_T)
        row["svcca_tofinal_heldout"] = sim.svcca(Phi_teSub, Phi_teSub_T, var_keep=args.svcca_var)
        if prev_sub is not None:
            row["cka_biased_consec"] = sim.linear_cka(Phi_sub, prev_sub)
            row["cka_debiased_consec"] = sim.linear_cka_debiased(Phi_sub, prev_sub)
        rows_sim.append(row)
        prev_sub = Phi_sub

        # ---- spectral directions for this checkpoint (cheap) ----
        V_t, s_t, mean_t = directions_via_covariance(load_feat(run_dir, "full_train", ep),
                                                     center=args.center)

        # ---- subspace dynamics per k (cheap) ----
        for k in args.ks:
            k = int(k)
            pa = sp.principal_angles_to_reference(V_t, V_T, k)
            # component-wise CKA-to-final on the cka subset, per-checkpoint basis vs final basis
            main_t, resid_t = sp.reconstruct_components(Phi_sub, V_t, k, mean=mean_t)
            main_T, resid_T = sp.reconstruct_components(Phi_sub_T, V_T, k, mean=mean_T)
            rows_sub.append(dict(
                tag, epoch=ep, k=k,
                sv_mass=sp.mass_at_rank(s_t, k, "singular"),
                energy_mass=sp.mass_at_rank(s_t, k, "energy"),
                pa_mean_cos=pa["mean_cos"], pa_min_cos=pa["min_cos"], pa_grassmann=pa["grassmann"],
                cka_main_tofinal=sim.linear_cka(main_t, main_T),
                cka_resid_tofinal=sim.linear_cka(resid_t, resid_T),
            ))

        # ---- Fisher per component (cheap; top components only) ----
        kf = int(args.fisher_k)
        fish = sp.fisher_ratio_per_component(Phi_tr_T if args.fisher_on_final else
                                             load_feat(run_dir, "full_train", ep),
                                             y_tr, V_t, kf, mean=mean_t)
        for j, fr in enumerate(fish):
            rows_fish.append(dict(tag, epoch=ep, component=j, fisher=float(fr),
                                  sing_val=float(s_t[j])))

        # ---- full-feature probe functional metrics + head-catchup (cheap-ish) ----
        Xtr = load_feat(run_dir, "full_train", ep)[tr_idx]
        Xte = Phi_teSub if args.func_on_subset else load_feat(run_dir, "full_test", ep)
        yte = y_te[te_idx] if args.func_on_subset else y_te
        fun = pe.fit_probe(Xtr, y_tr[tr_idx], Xte, yte, standardize=args.standardize)
        for nb in args.head_budgets:
            hc = pe.head_catchup_probe(Xtr, y_tr[tr_idx], Xte, yte, n_iter=int(nb),
                                       standardize=args.standardize)
            fun[f"head_acc_iter{int(nb)}"] = hc["head_acc"]
        rows_fun.append(dict(tag, epoch=ep, **fun))

        # ---- rank-accuracy sweep (EXPENSIVE; strided epochs) ----
        if (ei % args.rank_epoch_stride) == 0 or ep == T:
            Ztr_self = sp.project_topk(Xtr, V_t, max(args.ks), mean=mean_t)
            Zte_self = sp.project_topk(Xte, V_t, max(args.ks), mean=mean_t)
            Ztr_fin = sp.project_topk(Xtr, V_T, max(args.ks), mean=mean_T)
            Zte_fin = sp.project_topk(Xte, V_T, max(args.ks), mean=mean_T)
            self_res = pe.rank_sweep_probe(Ztr_self, y_tr[tr_idx], Zte_self, yte, args.ks,
                                           standardize=args.standardize)
            fin_res = pe.rank_sweep_probe(Ztr_fin, y_tr[tr_idx], Zte_fin, yte, args.ks,
                                          standardize=args.standardize)
            for r in self_res:
                rows_rank.append(dict(tag, epoch=ep, basis="self", **r))
            for r in fin_res:
                rows_rank.append(dict(tag, epoch=ep, basis="final", **r))

    out = os.path.join(run_dir, "results")
    os.makedirs(out, exist_ok=True)
    written = {}
    for name, rows in [("ext_similarity", rows_sim), ("ext_subspace", rows_sub),
                       ("ext_rank_accuracy", rows_rank), ("ext_fisher", rows_fish),
                       ("ext_probe_functional", rows_fun)]:
        if rows:
            df = pd.DataFrame(rows)
            path = os.path.join(out, f"{name}.csv")
            df.to_csv(path, index=False)
            written[name] = (path, len(df))
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/local/data/gme101")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="run dir names; default = all final_lr*_seed* under --base")
    ap.add_argument("--center", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--standardize", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--ks", type=int, nargs="*", default=DEFAULT_KS)
    ap.add_argument("--probe-subset", type=int, default=10000)
    ap.add_argument("--rank-epoch-stride", type=int, default=2)
    ap.add_argument("--fisher-k", type=int, default=64)
    ap.add_argument("--fisher-on-final", action="store_true",
                    help="compute Fisher in each checkpoint's basis but on FINAL features")
    ap.add_argument("--func-on-subset", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--head-budgets", type=int, nargs="*", default=[1, 3, 10])
    ap.add_argument("--svcca-var", type=float, default=0.99)
    ap.add_argument("--max-epochs", type=int, default=0)
    args = ap.parse_args()
    args.max_epochs = args.max_epochs or None

    if args.runs:
        run_dirs = [os.path.join(args.base, r) for r in args.runs]
    else:
        run_dirs = sorted(glob.glob(os.path.join(args.base, "final_lr*_seed*")))
        run_dirs = [r for r in run_dirs if os.path.isdir(r)]

    for rd in run_dirs:
        print(f"[run] {os.path.basename(rd)}")
        written = compute_run(rd, args)
        for name, (path, n) in written.items():
            print(f"   {name:22s} {n:5d} rows -> {path}")


if __name__ == "__main__":
    main()
