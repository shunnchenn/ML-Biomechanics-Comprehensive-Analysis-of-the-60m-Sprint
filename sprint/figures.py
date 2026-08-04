"""The three figures the analysis is for.

`phase_ribbon` is the interpretable one: every panel runs 0-100% of ground
contact then 0-100% of flight, so athletes with different contact and flight
times are compared at the same point of the same phase rather than at the same
fraction of a stride.
"""
from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from . import skeleton
from .features import CURVES
from .model import spm

FAST, SLOW, GREY = "#2166ac", "#c44e52", "#888888"
UNITS = dict.fromkeys(CURVES, "deg")


def _phase_axis(ax, n_pts=None):
    """Label the concatenated contact|flight axis and mark toe-off."""
    ax.axvline(C.N_CONTACT - 0.5, color="k", lw=1.0, ls="--", alpha=0.6)
    ax.set_xticks([0, C.N_CONTACT // 2, C.N_CONTACT - 1,
                   C.N_CONTACT + C.N_FLIGHT // 2, C.N_PHASE - 1])
    ax.set_xticklabels(["TD", "50%", "TO", "50%", "TD"], fontsize=8)
    ax.set_xlim(0, (n_pts or C.N_PHASE) - 1)
    ax.grid(alpha=0.25, lw=0.5)


def phase_ribbon(curves, fast, title, path, n_perm=5000, curve_names=CURVES):
    """Fast vs slow mean +/- 95% CI per kinematic curve, with SPM significance strips.

    curves : (n_subj, N_PHASE, n_curves)
    fast   : boolean mask, True for the fast group
    """
    n = len(curve_names)
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 1.55 * n), sharex=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)

    for i, (ax, name) in enumerate(zip(axes, curve_names, strict=False)):
        c = curves[:, :, i]
        for mask, col, lab in ((fast, FAST, "fast"), (~fast, SLOW, "slow")):
            g = c[mask]
            m = np.nanmean(g, axis=0)
            se = np.nanstd(g, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(g), axis=0))
            ax.plot(m, color=col, lw=1.8, label=lab, zorder=3)
            ax.fill_between(np.arange(len(m)), m - 1.96 * se, m + 1.96 * se,
                            color=col, alpha=0.18, lw=0)

        _, clusters = spm(c, fast, n_perm=n_perm)
        for a, b, _, p in clusters:
            if p < 0.05:
                ax.axvspan(a - 0.5, b + 0.5, color="#f0c419", alpha=0.25, lw=0, zorder=0)
                ax.text((a + b) / 2, ax.get_ylim()[1], f"p={p:.3f}", ha="center",
                        va="top", fontsize=7, color="#7a5c00")
        ax.set_ylabel(f"{name}\n({UNITS[name]})", fontsize=8)
        _phase_axis(ax, curves.shape[1])
        if i == 0:
            ax.legend(fontsize=8, ncol=2, loc="best", frameon=False)

    axes[-1].set_xlabel("0-100% ground contact  |  0-100% flight", fontsize=9)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def coefficient_plot(en, ci, lg, title, path, top=12):
    """Elastic Net betas with bootstrap CIs, beside the fast-vs-slow log-odds."""
    coef = en["coef"].head(top).iloc[::-1]
    ci = ci.set_index("feature").reindex(coef.feature)
    lg_c = lg["coef"].set_index("feature").reindex(coef.feature)
    y = np.arange(len(coef))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 0.42 * len(coef) + 2.2),
                                 sharey=True, constrained_layout=True)
    a1.barh(y, coef.beta, color=[FAST if b > 0 else SLOW for b in coef.beta], alpha=0.85)
    a1.hlines(y, ci.lo, ci.hi, color="k", lw=1.2)
    a1.axvline(0, color="k", lw=0.8)
    a1.set_yticks(y)
    a1.set_yticklabels(coef.feature, fontsize=8)
    a1.set_xlabel("Elastic Net beta per +1 SD  (bars: 95% bootstrap CI)", fontsize=9)
    a1.set_title(f"LOO-CV R$^2$ = {en['r2_loo']:.3f},  RMSE = {en['rmse']:.3f}", fontsize=9)

    a2.barh(y, lg_c.log_odds, color=[FAST if b > 0 else SLOW for b in lg_c.log_odds], alpha=0.85)
    a2.axvline(0, color="k", lw=0.8)
    a2.set_xlabel("Fast-vs-slow log-odds per +1 SD", fontsize=9)
    a2.set_title(f"LOO AUC = {lg['auc_loo']:.3f}  "
                 f"(n={lg['n_fast']} fast / {lg['n_slow']} slow)", fontsize=9)
    for a in (a1, a2):
        a.grid(alpha=0.25, axis="x", lw=0.5)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def kinogram(tensor, fast, title, path, pcts=(0, 25, 50, 75, 100, 150)):
    """Fast vs slow mean pose at fixed points of contact and flight.

    A percentage above 100 indexes into flight: 150 is mid-flight.
    """
    cols = [min(round(p / 100 * (C.N_CONTACT - 1)) if p <= 100 else
                C.N_CONTACT + round((p - 100) / 100 * (C.N_FLIGHT - 1)),
                C.N_PHASE - 1) for p in pcts]
    labels = [f"{p}% contact" if p <= 100 else f"{p - 100}% flight" for p in pcts]

    fig, axes = plt.subplots(1, len(cols), figsize=(2.5 * len(cols), 4.2),
                             constrained_layout=True)
    means = {k: np.nanmean(tensor[m], axis=0)
             for k, m in (("fast", fast), ("slow", ~fast))}
    for ax, col, lab in zip(np.atleast_1d(axes), cols, labels, strict=True):
        for k, colr in (("slow", SLOW), ("fast", FAST)):
            pts = means[k][col]
            if not np.isnan(pts).all():
                skeleton.draw(ax, pts - pts[C.MARKER_IDX["pelvis"]].mean(0) * [1, 1, 0],
                              color=colr, lw=1.4, alpha=0.85)
        ax.set_title(lab, fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(f"{title}   (blue = fast, red = slow)", fontsize=11, fontweight="bold")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
