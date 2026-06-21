"""
Representation-similarity measures, computed on ROW-ALIGNED feature matrices
(same samples, same order, at two checkpoints). Use the cka_subset (n=2040) which has
fixed indices across epochs.

Includes:
  * linear_cka            -- the standard (biased) linear CKA, feature-space form
  * linear_cka_debiased   -- unbiased HSIC_1 estimator (Song et al. 2012; the version
                             used by Nguyen et al. minibatch-CKA). Addresses the
                             finite-sample inflation of biased CKA at n/d ~ 4.
  * svcca                 -- GL-invariant subspace correlation. This is the
                             *matched-invariance* counterpart to a linear probe:
                             a probe's accuracy is invariant to any invertible linear
                             map of the features, and so is SVCCA, whereas CKA is only
                             invariant to orthogonal maps + isotropic scaling. The gap
                             between svcca-to-final and cka-to-final isolates how much of
                             the "stabilization lag" is just the GL-vs-orthogonal
                             invariance discrepancy.
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- CKA

def _feature_center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Biased linear CKA via the efficient feature-space form (no n x n matrices).

    CKA = ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F ||Yc^T Yc||_F),  Xc, Yc feature-centered.
    """
    Xc = _feature_center(X)
    Yc = _feature_center(Y)
    xty = Xc.T @ Yc
    xtx = Xc.T @ Xc
    yty = Yc.T @ Yc
    num = np.sum(xty ** 2)
    den = np.sqrt(np.sum(xtx ** 2)) * np.sqrt(np.sum(yty ** 2))
    return float(num / den) if den > 0 else 0.0


def _hsic1(K: np.ndarray, L: np.ndarray) -> float:
    """Unbiased HSIC estimator (Song et al. 2012). K, L are n x n Gram matrices."""
    n = K.shape[0]
    Kt = K.copy(); np.fill_diagonal(Kt, 0.0)
    Lt = L.copy(); np.fill_diagonal(Lt, 0.0)
    a = Kt.sum(axis=1)              # row sums
    b = Lt.sum(axis=1)
    tr = float(np.sum(Kt * Lt))    # tr(Kt Lt) for symmetric matrices
    term1 = tr
    term2 = (a.sum() * b.sum()) / ((n - 1) * (n - 2))
    term3 = (2.0 / (n - 2)) * float(a @ b)   # 1^T Kt Lt 1 = (Kt1).(Lt1)
    return (term1 + term2 - term3) / (n * (n - 3))


def linear_cka_debiased(X: np.ndarray, Y: np.ndarray) -> float:
    """Debiased linear CKA using HSIC_1. Forms n x n linear Gram matrices.

    Recommended for the headline CKA numbers given d=512, n=2040 (ratio ~4): the biased
    estimator drifts upward as d/n grows, so report this alongside the biased value.
    """
    Xc = _feature_center(X)
    Yc = _feature_center(Y)
    K = Xc @ Xc.T
    L = Yc @ Yc.T
    hxy = _hsic1(K, L)
    hxx = _hsic1(K, K)
    hyy = _hsic1(L, L)
    den = np.sqrt(max(hxx, 0.0) * max(hyy, 0.0))
    return float(hxy / den) if den > 0 else 0.0


# -------------------------------------------------------------------------- SVCCA

def svcca(X: np.ndarray, Y: np.ndarray, var_keep: float = 0.99) -> float:
    """SVCCA similarity between two row-aligned representations.

    Reduce each rep by SVD to the top directions explaining >= var_keep of variance,
    then take canonical correlations between the resulting orthonormal subspaces.
    Returns the mean canonical correlation in [0, 1].

    Implementation note: after SVD-reducing to orthonormal left singular vectors
    Ux[:, :kx], Uy[:, :ky], the canonical correlations equal the singular values of
    Ux[:, :kx]^T Uy[:, :ky] (CCA between orthonormal bases). This is exactly Raghu et
    al.'s SVCCA and equals the cosines of the principal angles between the two
    top-variance sample-subspaces.
    """
    Xc = _feature_center(X)
    Yc = _feature_center(Y)
    Ux, sx, _ = np.linalg.svd(Xc, full_matrices=False)
    Uy, sy, _ = np.linalg.svd(Yc, full_matrices=False)
    kx = _n_for_var(sx, var_keep)
    ky = _n_for_var(sy, var_keep)
    cross = Ux[:, :kx].T @ Uy[:, :ky]
    cc = np.linalg.svd(cross, compute_uv=False)
    cc = np.clip(cc, 0.0, 1.0)
    return float(cc.mean())


def _n_for_var(s: np.ndarray, var_keep: float) -> int:
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    return int(np.searchsorted(energy, var_keep) + 1)


# ------------------------------------------------------- convenience wrappers

def cka_pair(X: np.ndarray, Y: np.ndarray, debiased: bool = True) -> dict:
    """Both CKA estimators for a pair, as a dict (handy for tidy CSV rows)."""
    return {
        "cka_biased": linear_cka(X, Y),
        "cka_debiased": linear_cka_debiased(X, Y) if debiased else np.nan,
    }
