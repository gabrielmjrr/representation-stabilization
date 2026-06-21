"""
Spectral analysis of penultimate features.

All functions operate on a feature matrix Phi of shape (n_samples, d) with d=512.

Conventions / choices made explicit (these are the points the reviewers flagged):
  * `center`:  if True, subtract the per-feature mean before SVD. This makes the
               squared singular values equal to PCA explained variance, which is the
               correct basis for any "explained variance" language. Deng et al. use the
               *uncentered* matrix; we default to centered and expose the flag so the
               thesis can state which it uses and report both if needed.
  * sv-mass vs energy-mass: `rank_for_mass(..., kind="singular")` thresholds on
               sum(sigma)/sum(sigma); `kind="energy"` thresholds on
               sum(sigma^2)/sum(sigma^2) (= explained variance). The 0.80 singular-mass
               split in the current draft corresponds to ~0.997 energy, which is why
               main~=full was near-tautological. Report both.

Right singular vectors V (columns, each in R^d) are the directions in feature space;
"dominant subspace" = top-k columns of V. We never call these "discriminatory" without
a supervised diagnostic (see fisher_ratio_per_component).
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import subspace_angles


def svd_features(Phi: np.ndarray, center: bool = True):
    """Economy SVD of the feature matrix.

    Returns (U, s, Vt, mean) where Phi_c ~= U @ diag(s) @ Vt and Vt has shape (r, d).
    s are singular values (descending). Right singular vectors (feature-space
    directions) are the rows of Vt / columns of Vt.T.
    """
    mean = Phi.mean(axis=0, keepdims=True) if center else np.zeros((1, Phi.shape[1]), dtype=Phi.dtype)
    Xc = Phi - mean
    # full_matrices=False -> U:(n,r) s:(r,) Vt:(r,d), r=min(n,d)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return U, s, Vt, mean


def cumulative_mass(s: np.ndarray, kind: str = "singular") -> np.ndarray:
    """Cumulative fraction of (singular | energy) mass as a function of rank k.

    kind="singular": cumsum(s)/sum(s)
    kind="energy"  : cumsum(s^2)/sum(s^2)   (= cumulative explained variance if centered)
    Returns an array c of length len(s) where c[k-1] is the fraction captured by top-k.
    """
    if kind == "singular":
        w = s
    elif kind == "energy":
        w = s ** 2
    else:
        raise ValueError("kind must be 'singular' or 'energy'")
    total = w.sum()
    if total == 0:
        return np.zeros_like(w)
    return np.cumsum(w) / total


def rank_for_mass(s: np.ndarray, q: float, kind: str = "singular") -> int:
    """Smallest k such that the top-k directions capture >= q of the chosen mass."""
    c = cumulative_mass(s, kind=kind)
    k = int(np.searchsorted(c, q) + 1)
    return min(k, len(s))


def mass_at_rank(s: np.ndarray, k: int, kind: str = "singular") -> float:
    """Fraction of (singular|energy) mass captured by the top-k directions."""
    c = cumulative_mass(s, kind=kind)
    if k <= 0:
        return 0.0
    return float(c[min(k, len(c)) - 1])


def project_topk(Phi: np.ndarray, V: np.ndarray, k: int, mean: np.ndarray | None = None) -> np.ndarray:
    """Coordinates of (optionally centered) Phi in the top-k right-singular basis.

    V has shape (d, r) (columns = feature-space directions). Returns (n, k) coords
    Z = Phi_c @ V[:, :k]. Use these as probe inputs for rank-accuracy curves and as the
    fixed-final-basis projection (pass the FINAL checkpoint's V here).
    """
    Xc = Phi if mean is None else (Phi - mean)
    return Xc @ V[:, :k]


def reconstruct_components(Phi: np.ndarray, V: np.ndarray, k: int, mean: np.ndarray | None = None):
    """Split Phi into main (top-k) and residual reconstructions in the *given* basis V.

    Returns (Phi_main, Phi_resid) both in the original d-dim space (so CKA/probes can be
    applied directly). If mean is given, the centering is added back so the reconstructions
    live in the same space as Phi.
    """
    Xc = Phi if mean is None else (Phi - mean)
    Vk = V[:, :k]
    main_c = (Xc @ Vk) @ Vk.T
    resid_c = Xc - main_c
    if mean is not None:
        return main_c + mean, resid_c + mean
    return main_c, resid_c


def fisher_ratio_per_component(Phi: np.ndarray, y: np.ndarray, V: np.ndarray, k: int,
                               mean: np.ndarray | None = None) -> np.ndarray:
    """Between-class / within-class variance along each of the top-k singular directions.

    This is the supervised diagnostic that tests whether high-VARIANCE directions are
    also class-DISCRIMINATIVE (they need not be). Large Fisher ratio in the early
    components supports the "dominant subspace carries class info" reading; a flat
    profile would refute it. Also the cheap bridge to neural-collapse class geometry.
    Returns an array of length k (Fisher ratio per component, component order = V order).
    """
    k = min(int(k), V.shape[1])
    Z = project_topk(Phi, V, k, mean=mean)            # (n, k_eff)
    k = Z.shape[1]
    classes = np.unique(y)
    grand = Z.mean(axis=0)                              # (k,)
    between = np.zeros(k)
    within = np.zeros(k)
    for c in classes:
        Zc = Z[y == c]
        nc = Zc.shape[0]
        mc = Zc.mean(axis=0)
        between += nc * (mc - grand) ** 2
        within += ((Zc - mc) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        fisher = between / within
    return fisher


def principal_angles_to_reference(V_t: np.ndarray, V_ref: np.ndarray, k: int):
    """Principal angles between the top-k subspaces of two checkpoints.

    V_t, V_ref have shape (d, r) (columns = feature-space directions). Compares
    span(V_t[:, :k]) with span(V_ref[:, :k]). Returns a dict with:
      mean_cos      : mean canonical correlation (1.0 => subspaces identical)
      min_cos       : worst-aligned direction (cos of the largest principal angle)
      grassmann     : Grassmann (geodesic) distance sqrt(sum theta_i^2)
    Use V_ref = final checkpoint's V to ask "when does the dominant subspace lock in?".
    """
    A = V_t[:, :k]
    B = V_ref[:, :k]
    thetas = subspace_angles(A, B)          # ascending angles, length min(k, ...)
    cos = np.cos(thetas)
    return {
        "mean_cos": float(cos.mean()),
        "min_cos": float(cos.min()),
        "grassmann": float(np.sqrt((thetas ** 2).sum())),
    }


def effective_rank(s: np.ndarray) -> float:
    """Spectral (entropy) effective rank: exp(H(p)) with p_i = sigma_i / sum(sigma).

    A scalar summary of how spread the spectrum is; useful as a single-number companion
    to the rank-mass curves (and connects to the NTK effective-rank literature).
    """
    p = s / s.sum()
    p = p[p > 0]
    H = -(p * np.log(p)).sum()
    return float(np.exp(H))
