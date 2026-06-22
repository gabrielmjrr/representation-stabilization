"""
Frozen-feature probes with functional metrics beyond top-1 accuracy.

Why this exists (reviewer point common to both reviews): top-1 accuracy is bounded and
saturates early, so "sufficiency precedes stabilization" is partly a ceiling artifact.
Probe NLL and margin are functionals of representation GEOMETRY and keep evolving after
accuracy plateaus -- recording them lets you show whether late training is genuinely
refining the representation (margin/NLL still improving) rather than doing nothing.

Also:
  * rank_sweep_probe -- probe accuracy as a function of retained rank k -> the
    rank-accuracy curve that replaces the single 80%-singular-mass split.
  * head_catchup_probe -- a budget-MATCHED linear head, to control for "the online head
    was just under-optimized" when claiming the backbone is decodable early. If a head
    trained with a budget comparable to the online head still saturates early while CKA
    keeps moving, the result is robust to the head-lag objection.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)


def _standardize(X_train, X_test):
    sc = StandardScaler().fit(X_train)
    return sc.transform(X_train), sc.transform(X_test)


def _margin_from_decision(scores: np.ndarray, y: np.ndarray) -> float:
    """Mean multiclass margin: (true-class score) - (max other-class score)."""
    n = scores.shape[0]
    true_score = scores[np.arange(n), y]
    tmp = scores.copy()
    tmp[np.arange(n), y] = -np.inf
    other = tmp.max(axis=1)
    return float(np.mean(true_score - other))


def fit_probe(X_train, y_train, X_test, y_test, C: float = 1.0,
              max_iter: int = 3000, standardize: bool = True) -> dict:
    """Logistic-regression probe. Returns acc, NLL, and mean margin on the test set.

    Standardization is ON by default and is recorded -- note that standardizing breaks
    exact GL-invariance of the probe, so for the invariance argument keep this consistent
    (or report the raw-feature variant) and state the choice in Methods.
    """
    if standardize:
        X_train, X_test = _standardize(X_train, X_test)
    # lbfgs uses the multinomial loss for multiclass by default (all recent sklearn).
    clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    scores = clf.decision_function(X_test)
    if scores.ndim == 1:  # binary edge case
        scores = np.column_stack([-scores, scores])
    acc = float((proba.argmax(1) == y_test).mean())
    nll = float(log_loss(y_test, proba, labels=clf.classes_))
    margin = _margin_from_decision(scores, y_test)
    return {"test_acc": acc, "test_nll": nll, "test_margin": margin}


def rank_sweep_probe(Z_train, y_train, Z_test, y_test, ks, C: float = 1.0,
                     standardize: bool = True) -> list[dict]:
    """Probe accuracy/NLL/margin using the top-k projected coordinates, for each k in ks.

    Z_train/Z_test are coordinates in the top-K basis (n, K) from
    spectral.project_topk; slicing [:, :k] gives the top-k. Pass coordinates in the
    PER-CHECKPOINT basis for the rank-accuracy curve, or in the FINAL basis for the
    fixed-final-basis projection.
    """
    out = []
    for k in ks:
        k = int(k)
        res = fit_probe(Z_train[:, :k], y_train, Z_test[:, :k], y_test, C=C,
                        standardize=standardize)
        res["k"] = k
        out.append(res)
    return out


def head_catchup_probe(X_train, y_train, X_test, y_test, n_iter: int = 5,
                       eta0: float = 0.01, standardize: bool = True,
                       seed: int = 0) -> dict:
    """A linear softmax head trained with a CAPPED budget (n_iter passes), as a control.

    Uses SGDClassifier(log_loss) with a fixed, small number of epochs to mimic a head
    that has only had a limited amount of optimization -- contrast its accuracy with the
    fully-converged `fit_probe`. Report both vs final network accuracy: if the
    budget-limited head still reaches near-final accuracy early, the early decodability is
    not merely an artifact of unbounded probe optimization.

    `n_iter` is the modeling knob (the "budget"); document the choice. A principled
    setting matches it to the per-checkpoint optimisation the online head actually
    received between consecutive checkpoints.
    """
    if standardize:
        X_train, X_test = _standardize(X_train, X_test)
    clf = SGDClassifier(loss="log_loss", max_iter=n_iter, tol=None,
                        learning_rate="constant", eta0=eta0, random_state=seed)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    acc = float((proba.argmax(1) == y_test).mean())
    nll = float(log_loss(y_test, proba, labels=clf.classes_))
    return {"head_acc": acc, "head_nll": nll, "head_budget_iter": int(n_iter)}
