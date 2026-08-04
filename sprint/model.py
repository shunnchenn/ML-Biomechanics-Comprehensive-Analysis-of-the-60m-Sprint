"""Interpretable models: penalised regression, a fast-vs-slow contrast, and SPM.

With n around 30 the goal is a readable effect per feature, not a leaderboard of
algorithms. Every coefficient below is on standardised features, so it reads as
"per +1 SD of this feature, that much change in the target".

SHAP is computed on the linear model, where it is exact (beta * (x - xbar)).
Tree-model SHAP at this sample size would be noise dressed up as attribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C


def fpca(curves, n=5):
    """PCA over stacked per-participant curves: (n_subj, n_phase, n_curve) -> scores."""
    X = curves.reshape(len(curves), -1)
    X = np.nan_to_num(X - np.nanmean(X, axis=0), nan=0.0)
    p = PCA(n_components=min(n, len(X) - 1), random_state=C.SEED)
    return p.fit_transform(X), p


ALPHAS = np.logspace(-4, 1, 60)


def _pipe():
    return make_pipeline(
        StandardScaler(),
        ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0], alphas=ALPHAS, cv=5,
                     max_iter=20000, random_state=C.SEED))


def elasticnet(X, y, names):
    """LOO-CV R-squared plus standardised coefficients from a full-data refit.

    Scaling and alpha selection happen inside each CV fold, so the reported
    R-squared does not see the held-out athlete.
    """
    pred = cross_val_predict(_pipe(), X, y, cv=LeaveOneOut())
    fit = _pipe().fit(X, y)
    return dict(
        r2_loo=float(r2_score(y, pred)),
        rmse=float(np.sqrt(np.mean((y - pred) ** 2))),
        pred=pred,
        model=fit,
        coef=pd.DataFrame({"feature": names, "beta": fit[-1].coef_})
             .sort_values("beta", key=np.abs, ascending=False),
    )


def bootstrap_ci(X, y, names, fit, n_boot=2000, level=95):
    """Percentile bootstrap CIs on the standardised coefficients.

    The penalty is held at the values `fit` selected on the full data instead of
    being re-tuned per resample. A resample carries only ~63% unique athletes, so
    re-tuning picks a larger alpha and shrinks every coefficient — the interval
    would then sit systematically inside the point estimate it is meant to bracket.
    Fixing the penalty measures sampling variability at one model, which is what
    the interval is supposed to mean.

    Caveat that no amount of fixing removes: an L1-penalised coefficient has a
    point mass at exactly zero, so its bootstrap distribution is not centred on
    the full-data estimate and these percentile intervals do not have exact
    coverage. Read `selected_pct` — how often a feature survived the penalty at
    all — as the primary stability measure, and the interval as a width.
    """
    rng = np.random.default_rng(C.SEED)
    est = fit[-1]
    boot = make_pipeline(StandardScaler(),
                         ElasticNet(alpha=est.alpha_, l1_ratio=est.l1_ratio_,
                                    max_iter=20000))
    B = np.stack([boot.fit(X[i], y[i])[-1].coef_
                  for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))])
    lo, hi = np.percentile(B, [(100 - level) / 2, 100 - (100 - level) / 2], axis=0)
    return pd.DataFrame({"feature": names, "lo": lo, "hi": hi,
                         "selected_pct": (B != 0).mean(axis=0) * 100})


def logistic_tertile(X, y, names, q=None):
    """Penalised fast-vs-slow logistic contrast on the outer tertiles of y.

    Returns coefficients as log-odds per +1 SD, with LOO AUC. The middle tertile
    is dropped so the contrast is between groups that genuinely differ.
    """
    q = q or C.FAST_SLOW_Q
    lo, hi = np.quantile(y, [q, 1 - q])
    keep = (y <= lo) | (y >= hi)
    Xs, lab = X[keep], (y[keep] >= hi).astype(int)
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(penalty="l2", C=0.5, max_iter=5000))
    prob = cross_val_predict(pipe, Xs, lab, cv=LeaveOneOut(), method="predict_proba")[:, 1]
    fit = pipe.fit(Xs, lab)
    return dict(
        auc_loo=float(roc_auc_score(lab, prob)),
        n_fast=int(lab.sum()), n_slow=int((1 - lab).sum()),
        coef=pd.DataFrame({"feature": names, "log_odds": fit[-1].coef_[0]})
             .sort_values("log_odds", key=np.abs, ascending=False),
    )


def shap_linear(fit, X):
    """Exact SHAP values for the fitted linear pipeline: (n_subj, n_feature)."""
    import shap

    Xz = fit[:-1].transform(X)
    return shap.LinearExplainer(fit[-1], Xz).shap_values(Xz)


def spm(curves, group, n_perm=5000, alpha=0.05):
    """Cluster-based permutation test between two groups of 1-D curves.

    Point-wise two-sample t, thresholded, then contiguous suprathreshold clusters
    are compared against the permutation distribution of the largest cluster mass.
    This is the standard way to say "these groups differ over 30-60% of contact"
    without testing every phase point independently.
    """
    from scipy.stats import t as tdist

    g = np.asarray(group, bool)
    n1, n2 = int(g.sum()), int((~g).sum())
    tcrit = tdist.ppf(1 - alpha / 2, n1 + n2 - 2)

    def tstat(mask):
        a, b = curves[mask], curves[~mask]
        d = np.nanmean(a, 0) - np.nanmean(b, 0)
        s = np.sqrt(np.nanvar(a, 0, ddof=1) / n1 + np.nanvar(b, 0, ddof=1) / n2)
        return np.divide(d, s, out=np.zeros_like(d), where=s > 0)

    def clusters(t):
        sup = np.abs(t) > tcrit
        out, i = [], 0
        while i < len(sup):
            if sup[i]:
                j = i
                while j + 1 < len(sup) and sup[j + 1]:
                    j += 1
                out.append((i, j, float(np.abs(t[i:j + 1]).sum())))
                i = j
            i += 1
        return out

    obs = tstat(g)
    found = clusters(obs)
    rng = np.random.default_rng(C.SEED)
    null = np.array([max([c[2] for c in clusters(tstat(rng.permutation(g)))] or [0.0])
                     for _ in range(n_perm)])
    return obs, [(a, b, m, float((null >= m).mean())) for a, b, m in found]
