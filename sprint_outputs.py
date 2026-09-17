"""
sprint_outputs.py — sections 7 and 8: every figure, every MP4.

The analysis lives in `sprint_pipeline.py` and is walked step by step in
`sprint_pipeline.ipynb`. That notebook is where the testing happens — each stage
comes back as a DataFrame you can take the head and the shape of. So this half
has no reason to be interactive. It has one job:

    python sprint_outputs.py          # every figure, every athlete's MP4

`run_all_outputs()` does the same thing from a notebook or a REPL, and every
renderer below can still be called on its own.

    figures/                 the cohort-wide PNGs
    figures/Block Start/     one block-exit kinogram per athlete, all 30
    mp4s/                    the three group animations
    mp4s/Trial Runs/         one three-panel video per athlete, all 30

Anything produced once PER ATHLETE gets its own Title Case folder, so a set of
thirty stays together instead of burying the handful of figures that describe
the cohort as a whole. `cohort_dir()` is the one place that names them.

The three group animations are all top 3 against bottom 3, and they answer three
different questions: one stride at top speed, the blocks through 10 steps
(~16 m), and the whole run with the entire 62.5 m held in one fixed view.

Needs ffmpeg on PATH.

--------------------------------------------------------------------------
Why the animation code is written the way it is
--------------------------------------------------------------------------
The earlier renderers looked choppy for four separate reasons, and all four are
fixed here rather than papered over with a higher bitrate.

1. FRAME RATE.      Trials were rendered at 20 fps from every 2nd or 3rd sample
                    of a 60 Hz capture, so at best 20 distinct body positions
                    reached the eye each second. Now every captured sample is
                    drawn and the output runs at the capture rate — 60 fps, real
                    time. `every=N` still exists for a quick preview and drops
                    the frame rate to match, so timing stays honest.
2. LAYOUT JITTER.   The group animations used constrained layout with a title
                    that changed every frame. Constrained layout re-solves on
                    every draw, so the axes crept a pixel or two whenever the
                    title text changed width — the whole picture shivered. The
                    axes are now placed at fixed rectangles and the changing
                    text sits at a fixed position, so nothing can move.
3. CAMERA JITTER.   The side-on camera tracked the T12 marker, which oscillates
                    with every stride, so the background slid back and forth
                    under the athlete. The camera now follows a smoothed track.
4. LOOP SEAM.       A stride cycle is sampled 0-100 % inclusive, and 100 % is
                    the same instant as 0 % of the next cycle. Rendering all 101
                    samples therefore showed one frame twice at every loop join.
                    The last sample is now dropped, so the loop is seamless.

Rendering every frame instead of every third is 3x the frames, so the drawing
had to get cheaper to match. Two changes pay for it: the artists are built once
and updated per frame (`_Skeleton`, `set_data`) instead of clearing and
re-plotting the axes, and frames go to ffmpeg as raw RGBA straight off the Agg
canvas instead of being PNG-encoded and decoded again.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
# NOTE: the backend is deliberately left alone. Forcing "Agg" here renders the
# MP4 frames fine but silently stops figures displaying inline in a notebook.
# The video renderers below sidestep the question entirely: they build a bare
# Figure with an Agg canvas attached by hand and never touch pyplot.
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

import sprint_pipeline as SP
from sprint_pipeline import (
    # paths and cohort
    HERE, C3D_DIR, EXCLUDED_PIDS, SPRINT_DIST_M, PARTICIPANT_ANTHRO,
    # coordinate columns and marker indices
    _AP, _VERT, MK_PELVIS_ALL, MK_T12, MK_R_HEEL, MK_L_HEEL, MK_R_TOE,
    MK_L_TOE, THORAX_MKS_64, R_HEEL, L_HEEL,
    # colours
    COL_FAST, COL_SLOW, COL_BODY, COL_AXIAL, COL_RIGHT, COL_LEFT, COL_DOT,
    LEFT_ALPHA,
    # analysis
    ANGLE_NAMES, PAIRED_JOINTS, clean_for_angles, load_c3d,
    detect_sprint_start, height_scale_for, find_top_speed_strides,
    find_first_steps, stride_cycle_angles, stride_cycle_raw64,
    smooth_angle_curves, run_angle_fpca, fast_vs_slow_angles, limb_rom_order,
    build_feature_table, loo_predict, fpca_fold_transform,
    model_ladder, permutation_test, influence_report,
)

# ── Output directories ──────────────────────────────────────────────────────
# The MP4s get their own folder. The output run writes one per athlete, so
# thirty-odd videos would otherwise be mixed in with the PNGs.
FIG_OUT = HERE / "figures"; FIG_OUT.mkdir(exist_ok=True)
MP4_OUT = HERE / "mp4s";    MP4_OUT.mkdir(exist_ok=True)

# Anything produced once PER ATHLETE goes in its own Title Case subfolder, so a
# cohort-wide set stays together and never buries the handful of figures that
# describe the cohort as a whole.
def cohort_dir(parent, name):
    """`figures/Block Start/`-style folder for one all-30-athletes output set.

    Title Case, first letter of every word capitalised, created on demand.
    """
    out = Path(parent) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def cohort_pids():
    """Every participant ID with a trial on disk, excluded ones dropped."""
    seen = []
    for fpath in sorted(C3D_DIR.glob("*.c3d")):
        pid = fpath.stem.split("-")[0].strip()
        if pid not in EXCLUDED_PIDS and pid not in seen:
            seen.append(pid)
    return seen


# ==========================================================================
# 7 · Figures
# ==========================================================================

AXIAL_POLYS = [
    [2,3,5,4,2,6,7,3,7,8,6,8,9,10,11,14,16,21,12,15,11,15,21,15,22,15,12,22,16],
    [22,11], [21,11], [17,18,19,17,20,18,20,19,20,16],
]


RIGHT_POLYS = [
    [21,23,24,29,24,21], [21,29,27,29,28,27], [24,27,24,28,27], [23,27,23,28,27],
    [27,33,28,33,34,33,35,33,34,27],
    [39,40,39,41,39,42,41,40,42,47,40,47,41,47], [47,48,47,49],
    [40,48,40,49], [41,48,41,49],
    [48,49,53,48,53,54,53,55,53,56,57,56,58,53],
]


LEFT_POLYS = [
    [22,25,26,25,32,22], [22,26],
    [30,31,25,30,25,31,25,26,30,26,31,26], [32,30,31,30,32,31,32],
    [30,36,30,37,30,38,36,37,31,36,31,37,31,38,31],
    [43,44,43,45,43,46,45,44,46], [50,46,50,45,50,44,50], [50,51,50,52,50],
    [44,51,44,52,44], [45,51,45,52,45],
    [51,52,51,59,52,59,60,59,60,59,62,59,61,62,63,62,64],
]


def _z(polys):
    return [[i - 1 for i in poly] for poly in polys]


AXIAL_IDX = _z(AXIAL_POLYS)


RIGHT_IDX = _z(RIGHT_POLYS)


LEFT_IDX  = _z(LEFT_POLYS)


def _draw_raw64(ax, frame, color, alpha=0.95, lw=1.6, label=None, floor=None,
                left_alpha=0.35):
    """Draw one 64-marker skeleton in the sagittal plane, in METRES.

    Same bones as the MP4 exporter, so the anatomy in the figures and in the
    videos match. The LEFT limbs are drawn faint: in a sagittal view the two
    sides sit on top of each other and cross constantly, so without a contrast
    it is impossible to tell which leg is which.
    """
    pts = frame.copy()
    centre = pts[MK_PELVIS_ALL].mean(axis=0).copy()
    centre[_VERT] = 0.0
    pts = pts - centre                                    # centre horizontally
    if floor is not None:
        pts[:, _VERT] -= floor                            # one fixed floor

    # Trunk and right side solid, left side faded.
    first = True
    for polys, a_mul in ((AXIAL_IDX, 1.0), (RIGHT_IDX, 1.0), (LEFT_IDX, left_alpha)):
        for poly in polys:
            ax.plot(pts[poly, _AP], pts[poly, _VERT], color=color, lw=lw,
                    alpha=alpha * a_mul, solid_capstyle="round",
                    label=label if first else None)
            first = False


def pc_raw64_extremes(cycles_by_pid, fpca, pc=1, k=2.0):
    """Marker-space skeletons for mean +- k SD of a component.

    The component lives in ANGLE space. To draw it as a real skeleton, each
    marker coordinate at each phase is regressed on the participants' PC scores,
    and the fitted line is evaluated at +-k standard deviations. So this is a
    faithful projection of the angle component into marker space, not a second
    PCA — the analysis is still the one method, done once.

    Returns (minus, plus), each (n_points, 64, 3).
    """
    pids   = fpca["pids"]
    X      = np.stack([cycles_by_pid[p] for p in pids])       # (n_pid, pts, 64, 3)
    scores = fpca["scores"][:, pc - 1]

    # 1. Least-squares slope of every coordinate against the PC score.
    s_c   = scores - scores.mean()
    mean  = X.mean(axis=0)
    slope = np.tensordot(s_c, X - mean, axes=(0, 0)) / np.sum(s_c ** 2)

    # 2. Step out k standard deviations either side of the mean posture.
    step = k * float(scores.std())
    return mean - step * slope, mean + step * slope


def plot_pc_anatomy(cycles_by_pid, fpca, peak_vel, pc=1, k=2.0,
                    phases=(0, 25, 50, 75), title_extra="", fname=None,
                    save=True):
    """Draw a component as the REAL 64-marker skeleton, across the cycle.

    Same skeleton construction as the MP4s. Which side is faster is worked out
    from the sign of the component's correlation with peak velocity and written
    into the legend, so there is no guessing which colour is which.
    """
    minus, plus = pc_raw64_extremes(cycles_by_pid, fpca, pc, k)

    # 1. Work out which end of the component belongs to the faster athletes.
    #    A negative correlation means a HIGHER score goes with a LOWER speed.
    r = float(np.corrcoef(fpca["scores"][:, pc - 1],
                          np.array([peak_vel[p] for p in fpca["pids"]]))[0, 1])
    plus_lbl  = "faster" if r > 0 else "slower"
    minus_lbl = "slower" if r > 0 else "faster"

    # Blue is always the faster end, red always the slower one. Colouring by the
    # +/- sign of the component instead would make red the faster side whenever
    # a component correlates negatively with speed - which top-speed PC1 does.
    plus_col  = COL_FAST if plus_lbl  == "faster" else COL_SLOW
    minus_col = COL_FAST if minus_lbl == "faster" else COL_SLOW

    # 2. One floor for every panel, taken across both skeletons, so the figures
    #    sit on the same ground instead of each floating to its own.
    foot_ids = [MK_R_HEEL, MK_L_HEEL, MK_R_TOE, MK_L_TOE]
    floor = min(float(minus[:, foot_ids, _VERT].min()),
                float(plus[:,  foot_ids, _VERT].min()))

    fig, axes = plt.subplots(1, len(phases), figsize=(4.1 * len(phases), 6.8),
                             constrained_layout=True, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ph in zip(axes, phases):
        _draw_raw64(ax, minus[ph], minus_col, floor=floor,
                    label=f"mean − {k:g}·PC{pc}  ({minus_lbl})")
        _draw_raw64(ax, plus[ph],  plus_col, floor=floor,
                    label=f"mean + {k:g}·PC{pc}  ({plus_lbl})")
        ax.axhline(0.0, color="#8a6d3b", lw=1.4, alpha=0.6, zorder=0)
        ax.set_title(f"{ph} % of cycle", fontsize=11)
        ax.set_aspect("equal"); ax.grid(alpha=0.22)
        ax.set_xlabel("fore-aft (m)")
    axes[0].set_ylabel("vertical (m)")

    imp  = np.linalg.norm(fpca["loadings"][pc - 1], axis=0)
    top3 = [fpca["names"][i] for i in np.argsort(-imp)[:3]]
    fig.suptitle(
        f"PC{pc} on the real skeleton{title_extra} — "
        f"{fpca['explained_var'][pc - 1]:.1f} % of variance   "
        f"(r = {r:+.2f} with peak velocity)",
        fontweight="bold", fontsize=12)
    # The "which joints" note rides as the legend TITLE. A separate fig.text in
    # the bottom margin lands on top of the legend, because constrained layout
    # puts both in the same strip; as the title it gets its own line.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2,
               frameon=False, fontsize=11,
               title=f"mainly: {', '.join(top3)}   |   "
                     f"thin pale limbs are the LEFT side",
               title_fontproperties={"style": "italic", "size": 10.5})

    if save:
        out = FIG_OUT / (fname or f"pc{pc}_anatomy.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        return out      # left open on purpose: a notebook then shows it once
    return fig


def plot_pc_loading_ranking(fpca, pc=1, save=True):
    """Which joints does this component actually use? A ranked bar chart.

    Importance for one angle is the size (L2 norm) of that angle's loading
    across the whole cycle. A long bar means the component varies a lot in that
    joint; a short bar means the joint barely participates.
    """
    k          = pc - 1
    loading    = fpca["loadings"][k]                 # (points, angles)
    importance = np.linalg.norm(loading, axis=0)
    order      = np.argsort(importance)              # ascending, so biggest ends on top
    names      = [fpca["names"][i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.barh(range(len(order)), importance[order], color="#4c72b0")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names)
    ax.set_xlabel("loading magnitude (unitless)")
    ax.set_title(f"PC{pc} — which joints carry this component\n"
                 f"{fpca['explained_var'][k]:.1f} % of variance",
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    if save:
        out = FIG_OUT / f"pc{pc}_loading_ranking.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        return out      # left open on purpose: a notebook then shows it once
    return fig


ANGLE_PAIRS = [("hip", "hip_R", "hip_L"), ("knee", "knee_R", "knee_L"),
               ("ankle", "ankle_R", "ankle_L"),
               ("shoulder", "shoulder_R", "shoulder_L"),
               ("elbow", "elbow_R", "elbow_L"),
               ("trunk lean", "trunk_lean", None),
               ("thigh separation", "thigh_separation", None),
               ("arm separation", "arm_separation", None)]


def plot_fast_vs_slow(cohort, peak_vel, n=5, save=True):
    """Angle curves through the cycle, fast group against slow group.

    Eight panels, one per joint. Right side solid, left side dotted and shifted
    half a cycle so the two lie on top of each other - any remaining gap is a
    genuine left-right asymmetry rather than the legs simply alternating.
    """
    _table, fast, slow, fast_ids, slow_ids = fast_vs_slow_angles(cohort, peak_vel, n)
    x = np.linspace(0, 100, fast.shape[0])

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for ax, (title, r_name, l_name) in zip(axes.ravel(), ANGLE_PAIRS):
        for name, style, shift in ((r_name, "-", 0), (l_name, ":", 50)):
            if name is None:
                continue
            j = ANGLE_NAMES.index(name)
            # The left leg does the SAME thing half a cycle later, so shifting it
            # by 50 % lays it on top of the right. (Flipping it vertically only
            # looks right because these curves are near-sinusoidal - it gives
            # r = 0.42 at the ankle where the shift gives 1.00.) Any gap that
            # survives the shift is a real left-right asymmetry.
            s = np.roll(slow[:, j], shift)
            f = np.roll(fast[:, j], shift)
            # Rolling wraps the array, which leaves a seam where the end meets
            # the start. Draw the two halves as separate lines so no stroke is
            # drawn across that join.
            cuts = [slice(None)] if shift == 0 else [slice(0, shift), slice(shift, None)]
            for k, sl in enumerate(cuts):
                lbl = "" if k else ("" if shift == 0 else "  (left, +50%)")
                ax.plot(x[sl], s[sl], color=COL_SLOW, lw=2, ls=style,
                        label=("5 slowest" + lbl) if k == 0 else None)
                ax.plot(x[sl], f[sl], color=COL_FAST, lw=2, ls=style,
                        label=("5 fastest" + lbl) if k == 0 else None)
            if shift == 0:
                ax.fill_between(x, s, f, color="#999999", alpha=0.25)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("% of stride cycle", fontsize=9)
        ax.set_ylabel("degrees", fontsize=9)
        ax.tick_params(labelsize=8); ax.grid(alpha=0.25)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2,
               frameon=False, fontsize=11)
    fig.suptitle(
        f"Joint angles through the stride cycle — {n} fastest vs {n} slowest\n"
        f"solid = right, dotted = left shifted +50% of the cycle    |    "
        f"fast: {', '.join(fast_ids)}   slow: {', '.join(slow_ids)}",
        fontweight="bold", fontsize=12)

    if save:
        out = FIG_OUT / "fast_vs_slow_angles.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        return out      # left open on purpose: a notebook then shows it once
    return fig


def plot_block_exit_steps(trial, save=True):
    """Block exit and the first three steps, as five sagittal skeletons.

    Reads left to right: the athlete set in the blocks, the moment the hands
    leave, then each of the first three foot contacts. This is the phase the
    old velocity-only start was cutting off.

    Left limbs are faint, same convention as every other skeleton figure here.
    """
    t   = clean_for_angles(trial)
    raw = t["raw64"]

    # 1. Frame 0 IS the set position now, because clean_for_angles starts there.
    #    Block exit is the FRONT foot leaving - the last contact with the blocks.
    #    The hands lift first (about 4 frames in) but that is barely a change of
    #    posture; the front toe leaving is the moment the athlete is airborne
    #    from the blocks, and it lands between the hands lifting and step 1.
    #    25 mm of rise is the same threshold the older kinogram code settled on.
    front_toe = (MK_R_TOE if raw[0, MK_R_HEEL, _AP] > raw[0, MK_L_HEEL, _AP]
                 else MK_L_TOE)
    toe_z = raw[:, front_toe, _VERT]
    off   = np.where(toe_z > toe_z[0] + 0.025)[0]
    exit_f = int(off[0]) if len(off) else 1

    # 2. The first three steps, from the detector used by the analysis.
    steps, _ = find_first_steps(t["mk"], n_steps=3)
    cols = [("Set — all four down", 0), ("Block exit (front foot)", exit_f)]
    cols += [(f"Step {i+1} contact", a) for i, (a, _b) in enumerate(steps)]

    # 3. One floor for every panel so the skeletons share a ground line.
    foot_ids = [MK_R_HEEL, MK_L_HEEL, MK_R_TOE, MK_L_TOE]
    floor = float(raw[[f for _n, f in cols]][:, foot_ids, _VERT].min())

    fig, axes = plt.subplots(1, len(cols), figsize=(3.5 * len(cols), 5.6),
                             constrained_layout=True, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, f) in zip(axes, cols):
        _draw_raw64(ax, raw[f], "#1f4e79", floor=floor, lw=1.7)
        ax.axhline(0.0, color="#8a6d3b", lw=1.4, alpha=0.6, zorder=0)
        ax.set_title(f"{name}\nframe {f}  ({f / t['fs']:.2f} s)", fontsize=10)
        ax.set_aspect("equal"); ax.grid(alpha=0.22)
        ax.set_xlabel("fore-aft (m)")
    axes[0].set_ylabel("vertical (m)")
    fig.suptitle(f"{t['pid']} — block exit and the first three steps"
                 f"   (peak {t['peak_vel_ms']:.2f} m/s)\n"
                 f"solid = right side, faint = left side",
                 fontweight="bold", fontsize=12)

    if save:
        # One per athlete, so it goes in the cohort folder rather than loose
        # among the figures that describe the whole cohort.
        out = cohort_dir(FIG_OUT, "Block Start") / f"block_exit_{t['pid']}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        return out      # left open on purpose: a notebook then shows it once
    return fig


# --------------------------------------------------------------------------
# Figures for the ML stage
# --------------------------------------------------------------------------

def plot_velocity_fit(feat, pid=None, save=True):
    """The target being cleaned: raw numerical derivative against its B-spline."""
    pid = pid or sorted(feat["vel"])[0]
    v_raw, v_sm, fs = feat["vel"][pid]
    t = np.arange(len(v_raw)) / fs

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.7, 1]})

    ax = axes[0]
    ax.plot(t, v_raw, color="#999999", lw=0.9, label="raw (numerical derivative)")
    ax.plot(t, v_sm, color=COL_FAST, lw=2.2, label="penalised B-spline fit")
    ax.axhline(v_raw.max(), color="#999999", ls=":", lw=1)
    ax.axhline(v_sm.max(),  color=COL_FAST, ls=":", lw=1)
    ax.annotate(f"{v_raw.max() - v_sm.max():+.2f} m/s",
                xy=(t[-1] * 0.72, (v_raw.max() + v_sm.max()) / 2),
                fontsize=10, fontweight="bold", color=COL_SLOW)
    ax.set_xlabel("time (s)"); ax.set_ylabel("speed (m/s)")
    ax.set_title(f"{pid} — the peak is read off the curve, not the samples",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25)

    ax = axes[1]
    infl = (feat["y_raw"] - feat["y"]).values
    ax.hist(infl, bins=12, color=COL_SLOW, alpha=0.75, edgecolor="white")
    ax.axvline(infl.mean(), color="k", ls="--", lw=1.5,
               label=f"mean {infl.mean():+.2f} m/s")
    ax.set_xlabel("noise removed from peak (m/s)"); ax.set_ylabel("athletes")
    ax.set_title("Every athlete's peak was too high", fontsize=11,
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25)

    if save:
        out = FIG_OUT / "velocity_bspline_fit.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        return out
    return fig


def plot_limb_asymmetry(feat, save=True):
    """Left-right ROM difference against the noise floor that would produce it."""
    floors = feat["floors"]
    pids   = [p for p in sorted(floors) if floors[p]]
    between = {j: np.array([floors[p][j]["between_limb"] for p in pids])
               for j in PAIRED_JOINTS}
    within  = {j: np.array([floors[p][j]["within_limb"] for p in pids])
               for j in PAIRED_JOINTS}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)

    ax = axes[0]
    xs = np.arange(len(PAIRED_JOINTS))
    ax.bar(xs - 0.2, [between[j].mean() for j in PAIRED_JOINTS], 0.4,
           color=COL_FAST, label="between limbs (the asymmetry)")
    ax.bar(xs + 0.2, [within[j].mean() for j in PAIRED_JOINTS], 0.4,
           color="#999999", label="within limb, stride to stride (noise floor)")
    ax.set_xticks(xs); ax.set_xticklabels(PAIRED_JOINTS)
    ax.set_ylabel("ROM difference (deg)")
    ax.set_title("Is the asymmetry bigger than the measurement?",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    hi_is_left = np.array([[limb_rom_order(feat["cycles"][p])[j]
                            for j in PAIRED_JOINTS] for p in pids])
    ax.bar(xs, hi_is_left.mean(axis=0) * 100, 0.55, color=COL_BODY)
    ax.axhline(50, color=COL_SLOW, ls="--", lw=1.5, label="chance")
    ax.set_xticks(xs); ax.set_xticklabels(PAIRED_JOINTS)
    ax.set_ylabel("% of athletes where LEFT is the larger-ROM limb")
    ax.set_ylim(0, 100)
    ax.set_title("No side is systematically dominant", fontsize=11,
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25, axis="y")

    if save:
        out = FIG_OUT / "limb_rom_asymmetry.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        return out
    return fig


def plot_model_comparison(table, null_r2=None, save=True):
    """The ladder as a chart, with the honest zero line drawn in."""
    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(t) + 2.2),
                           constrained_layout=True)
    colours = [COL_FAST if v > 0 else COL_SLOW for v in t["LOO_R2"]]
    ax.barh(t["model"], t["LOO_R2"], color=colours, height=0.62)
    if null_r2 is not None:
        ax.axvline(null_r2, color="k", ls="--", lw=1.5,
                   label=f"intercept-only ({null_r2:+.3f})")
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.axvline(0, color="#444444", lw=1)
    # Leave room on both sides for the value labels, which sit outside the bar
    # ends -- without it the longest bar's number lands on top of the y tick
    # label on the left, or falls off the axis on the right.
    lo, hi = float(t["LOO_R2"].min()), float(t["LOO_R2"].max())
    span   = max(hi - lo, 0.1)
    pad    = 0.028 * span
    ax.set_xlim(min(lo, 0) - 0.26 * span, max(hi, 0) + 0.22 * span)
    for i, v in enumerate(t["LOO_R2"]):
        ax.text(v + (pad if v >= 0 else -pad), i, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=9, fontweight="bold")
    ax.set_xlabel("leave-one-out $R^2$   (negative = worse than guessing the mean)")
    ax.set_title("Predicting peak velocity from kinematics",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25, axis="x")
    if save:
        out = FIG_OUT / "model_comparison_loocv.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        return out
    return fig


def plot_diagnostics(y, pred, X, pids, null_dist=None, obs_r2=None, save=True):
    """Four panels: is the fit real, and is any one athlete carrying it?"""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    resid = y - pred
    inf   = influence_report(X, y, pids)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3), constrained_layout=True)

    ax = axes[0]
    fast = y >= np.median(y)
    ax.scatter(pred[fast], y[fast], s=55, color=COL_FAST, edgecolor="white", zorder=3)
    ax.scatter(pred[~fast], y[~fast], s=55, color=COL_SLOW, edgecolor="white", zorder=3)
    lim = [min(y.min(), pred.min()) - 0.1, max(y.max(), pred.max()) + 0.1]
    ax.plot(lim, lim, "k--", lw=1.2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("predicted (held out, m/s)"); ax.set_ylabel("observed (m/s)")
    ax.set_title("Observed vs predicted", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.scatter(pred, resid, s=55, color=COL_BODY, edgecolor="white", zorder=3)
    ax.axhline(0, color="k", ls="--", lw=1.2)
    ax.set_xlabel("predicted (m/s)"); ax.set_ylabel("residual (m/s)")
    ax.set_title("Residuals vs fitted", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25)

    ax = axes[2]
    thr = 4.0 / len(y)
    top = inf.head(len(inf))
    cols = [COL_SLOW if d > thr else COL_BODY for d in top["cooks_D"]]
    ax.bar(range(len(top)), top["cooks_D"], color=cols)
    ax.axhline(thr, color=COL_SLOW, ls="--", lw=1.5, label=f"4/n = {thr:.3f}")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top["pid"], rotation=90, fontsize=6)
    ax.set_ylabel("Cook's distance")
    ax.set_title("Is one athlete carrying it?", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25, axis="y")

    ax = axes[3]
    if null_dist is not None:
        ax.hist(null_dist, bins=40, color="#999999", edgecolor="white")
        ax.axvline(obs_r2, color=COL_FAST, lw=2.5,
                   label=f"observed {obs_r2:+.3f}")
        p = (np.sum(null_dist >= obs_r2) + 1) / (len(null_dist) + 1)
        ax.set_title(f"Permutation null — p = {p:.4f}", fontsize=11,
                     fontweight="bold")
        ax.set_xlabel("LOO $R^2$ with the target shuffled")
        ax.set_ylabel("shuffles"); ax.legend(frameon=False, fontsize=9)
    else:
        ax.axis("off")
    ax.grid(alpha=0.25)

    if save:
        out = FIG_OUT / "model_diagnostics.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        return out
    return fig


# ==========================================================================
# 8 · Animation
# ==========================================================================

# ── The frame factory ───────────────────────────────────────────────────────

def _agg_figure(figsize, dpi):
    """A figure with an Agg canvas attached by hand — pyplot never sees it.

    Two reasons. It cannot disturb whatever backend a notebook is using, and it
    gives direct access to the RGBA buffer, which is what makes `_MP4Writer`
    cheap. Nothing has to be closed afterwards either: with no pyplot registry
    holding a reference, the figure is collected like any other object.
    """
    fig = Figure(figsize=figsize, dpi=dpi, facecolor="white")
    FigureCanvasAgg(fig)
    return fig


class _MP4Writer:
    """Raw RGBA frames straight from the Agg canvas into ffmpeg.

    The old writer saved every frame as a PNG in memory and handed ffmpeg the
    compressed bytes, which then had to decode them again. Compressing a frame
    only to decompress it one pipe later costs about 60 ms a frame and buys
    nothing, and at 60 fps that alone was most of the render time. Raw frames
    skip both halves.
    """

    def __init__(self, fig, out_path, fps):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is not on PATH — needed to write MP4s")
        self.fig = fig
        self.path = Path(out_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Draw once to learn the exact buffer size. Taking it from the buffer
        # rather than from figsize*dpi keeps this correct whatever rounding
        # matplotlib does.
        fig.canvas.draw()
        h, w, _ = np.asarray(fig.canvas.buffer_rgba()).shape

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
            "-framerate", str(fps), "-i", "-",
            "-an",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-preset", "veryfast",
            "-vf", "scale=ceil(iw/2)*2:ceil(ih/2)*2",   # yuv420p needs even dims
            "-r", str(fps),
            "-g", str(max(int(fps) * 2, 2)),
            "-movflags", "+faststart",                  # starts playing sooner
            str(self.path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.n_frames = 0

    def add(self):
        """Render the current state of the figure as one frame."""
        self.fig.canvas.draw()
        try:
            self.proc.stdin.write(self.fig.canvas.buffer_rgba())
        except BrokenPipeError:
            pass          # ffmpeg died; close() reports the return code
        self.n_frames += 1

    def close(self):
        """Always reached, even if a frame fails - the original leaked the process."""
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited {rc} writing {self.path.name}")
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class _Skeleton:
    """One skeleton's worth of Line2D artists, built once and moved per frame.

    Every renderer used to clear its axes and re-plot ~25 polylines each frame.
    Clearing throws away the artists, the axis limits, the grid and the labels,
    and all of it is rebuilt from scratch — which is both slow and the reason
    anything else on the axes had to be redrawn too. Here the lines are created
    once and only their data changes.
    """

    def __init__(self, ax, axial_c, right_c, left_c, lw=1.2,
                 left_alpha=1.0, dots=False, zorder=2):
        self.lines = []
        for polys, colour, alpha in ((AXIAL_IDX, axial_c, 1.0),
                                     (RIGHT_IDX, right_c, 1.0),
                                     (LEFT_IDX,  left_c,  left_alpha)):
            for poly in polys:
                line, = ax.plot([], [], color=colour, lw=lw, alpha=alpha,
                                solid_capstyle="round", zorder=zorder)
                self.lines.append((line, poly))
        self.dots = None
        if dots:
            self.dots = ax.scatter([], [], c=COL_DOT, s=10, linewidths=0,
                                   zorder=zorder + 1)

    def set_frame(self, pts, h=_AP, v=_VERT):
        """Move the skeleton to one frame of 64 marker positions."""
        for line, poly in self.lines:
            line.set_data(pts[poly, h], pts[poly, v])
        if self.dots is not None:
            self.dots.set_offsets(np.column_stack([pts[:, h], pts[:, v]]))


def _smooth_track(x, fs, win_s=0.5):
    """Centred moving average, for camera tracks only.

    The camera followed a single marker, and every marker on a sprinter
    oscillates once per stride. Panning with that oscillation slides the
    background back and forth under the athlete about three times a second,
    which reads as judder even though the skeleton itself is perfectly smooth.
    Averaging over half a second leaves the travel and removes the wobble.

    This is for the VIEW only. No measured quantity is smoothed here.
    """
    w = max(int(round(win_s * fs)) | 1, 3)          # odd, so it stays centred
    pad = w // 2
    return np.convolve(np.pad(x, pad, mode="edge"), np.ones(w) / w, mode="valid")


# ── One trial, three panels ─────────────────────────────────────────────────

def detrend_and_crop(pts: np.ndarray) -> np.ndarray:
    """Replicate the MATLAB pipeline:
       1. Reset origin to the most-posterior heel at frame 0
       2. Detrend Y as residual of linear fit y(x)
       3. Re-express X as distance along the fit line
       4. Crop at SPRINT_DIST_M forward distance
    """
    # ── 1. Origin reset ───────────────────────────────────────────────────
    f0     = pts[0]
    origin = f0[R_HEEL] if f0[R_HEEL, 0] < f0[L_HEEL, 0] else f0[L_HEEL]
    pts    = pts - origin

    n_frames, n_markers, _ = pts.shape
    x_all = pts[:, :, 0].ravel()
    y_all = pts[:, :, 1].ravel()

    # ── 2. Detrend Y as residual from linear fit ──────────────────────────
    p          = np.polyfit(x_all, y_all, 1)
    y_residual = y_all - np.polyval(p, x_all)

    # ── 3. Re-express X as distance along the fit line ─────────────────────
    line_y = np.polyval(p, x_all)
    x_new  = np.sqrt(x_all ** 2 + line_y ** 2)

    pts[:, :, 0] = x_new.reshape(n_frames, n_markers)
    pts[:, :, 1] = y_residual.reshape(n_frames, n_markers)

    # ── 4. Crop at 62.5 m forward ────────────────────────────────────────
    over = np.where(pts[:, :, 0] > SPRINT_DIST_M)
    if over[0].size:
        stop = over[0].min()
        pts  = pts[: stop + 1]

    return pts


def compute_thoracic_velocity(pts: np.ndarray, fs: float) -> np.ndarray:
    """3-D speed of the 7-marker thoracic centroid, m/s."""
    dt   = 1.0 / fs
    thor = pts[:, THORAX_MKS_64, :].mean(axis=1)
    vx   = np.gradient(thor[:, 0], dt)
    vy   = np.gradient(thor[:, 1], dt)
    vz   = np.gradient(thor[:, 2], dt)
    return np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)         # m/s


def render_video(c3d_path, out_path, fps=None, dpi=100, every=1) -> str:
    """Three-panel animation of one trial, written to `out_path` as MP4.

    `every`=N renders every Nth captured frame. It defaults to 1 — EVERY sample
    — because skipping frames was the main reason these videos looked choppy: a
    60 Hz capture rendered every 3rd frame shows 20 body positions a second.

    `fps` defaults to `fs / every`, so playback is real time whatever `every`
    is set to. Leave it alone unless you want deliberate slow motion.
    """
    pid           = Path(str(c3d_path)).stem
    _seg, pts, fs = load_c3d(c3d_path)   # this renderer draws the 64 raw markers
    pts           = detrend_and_crop(pts)
    start_frame   = detect_sprint_start(pts, fs)
    if start_frame > 0:
        pts = pts[start_frame:]
    vel      = compute_thoracic_velocity(pts, fs)            # m/s
    n_frames = len(pts)
    time     = np.arange(n_frames) / fs
    fps      = float(fs) / every if fps is None else fps

    max_vel_mps = float(vel.max())
    peak_frame  = int(np.argmax(vel))
    peak_time   = time[peak_frame]

    t12_x       = pts[:, MK_T12, 0]
    fin_idx     = np.where(t12_x > SPRINT_DIST_M)[0]
    finish_time = time[fin_idx[0]] if fin_idx.size else time[-1]

    # The side-on camera follows a SMOOTHED T12 track. T12 sways forward and
    # back within every stride, so panning with the raw marker drags the grid
    # and the ground back and forth under a runner who is in fact moving evenly.
    cam_x = _smooth_track(t12_x, fs, win_s=0.5)

    # Trail colours: thoracic speed through the jet map, one colour per segment.
    traj     = pts[:, MK_T12, :2]
    segs     = np.stack([traj[:-1], traj[1:]], axis=1)       # (n-1, 2, 2)
    seg_rgba = cm.jet(vel[:-1] / max(max_vel_mps, 1e-9))     # (n-1, 4)
    y_lim_path = (float(pts[:, :, 1].min()), float(pts[:, :, 1].max()))
    lane_y     = max(abs(y_lim_path[0]), abs(y_lim_path[1])) * 1.2

    fig = _agg_figure((12.78, 6.42), dpi)
    ax_vel  = fig.add_axes([0.06, 0.56, 0.42, 0.35])
    ax_side = fig.add_axes([0.55, 0.56, 0.42, 0.35])
    ax_path = fig.add_axes([0.05, 0.10, 0.92, 0.32])
    fig.suptitle(f"{pid}  •  Max V = {max_vel_mps:.2f} m/s  •  "
                 f"Time to 62.5 m = {finish_time:.2f} s",
                 fontsize=12, fontweight="bold", y=0.97)

    # ── Everything that never changes is drawn ONCE ──────────────────────────
    # Panel 1: velocity curve
    ax_vel.plot(time, vel, color="k", lw=1.4)
    ax_vel.scatter([peak_time], [vel[peak_frame]],
                   c=COL_RIGHT, s=80, edgecolors="k", zorder=5)
    now_dot, = ax_vel.plot([], [], "o", c="#1f77b4", ms=9,
                           markeredgecolor="k", zorder=6)
    info = ax_vel.text(0.98, 0.04, "", transform=ax_vel.transAxes,
                       fontsize=9, ha="right", va="bottom",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                 edgecolor="#888888", lw=0.6, alpha=0.9))
    ax_vel.set_xlabel("Time (sec)", fontsize=9)
    ax_vel.set_ylabel("Thoracic Velocity (m/s)", fontsize=9)
    ax_vel.set_xlim(0, time.max())
    ax_vel.set_ylim(0, vel.max() * 1.08)
    ax_vel.tick_params(labelsize=8)
    ax_vel.grid(True, alpha=0.3)

    # Panel 2: sagittal skeleton, camera-tracked
    skel = _Skeleton(ax_side, COL_AXIAL, COL_RIGHT, COL_LEFT, lw=1.2, dots=True)
    ax_side.set_ylim(pts[:, :, 2].min() - 0.05, pts[:, :, 2].max() + 0.05)
    ax_side.set_aspect("equal", adjustable="box")
    ax_side.set_xlabel("X (m)", fontsize=9)
    ax_side.set_ylabel("Z (m)", fontsize=9)
    ax_side.tick_params(labelsize=8)
    ax_side.grid(True, alpha=0.3)

    # Panel 3: top-down track
    ax_path.axhspan(-lane_y, lane_y, color="#f5f5f5", zorder=0)
    ax_path.axvline(0, color="#888888", lw=1, ls="--")
    ax_path.axvline(SPRINT_DIST_M, color="k", lw=2)
    ax_path.text(SPRINT_DIST_M, y_lim_path[1] * 0.85, " 62.5 m",
                 fontsize=9, fontweight="bold")
    # The whole trail is one LineCollection, built once. Only the alpha column
    # changes per frame, which is what makes it grow behind the athlete without
    # re-drawing hundreds of little line segments every time.
    trail = LineCollection(segs, linewidths=2.5, capstyle="round", zorder=2)
    trail_rgba = seg_rgba.copy()
    trail_rgba[:, 3] = 0.0
    trail.set_color(trail_rgba)
    ax_path.add_collection(trail)
    peak_dot, = ax_path.plot([], [], "o", c=COL_RIGHT, ms=11,
                             markeredgecolor="k", zorder=10)
    path_dot, = ax_path.plot([], [], "o", c="#1f77b4", ms=9,
                             markeredgecolor="k", zorder=11)
    ax_path.set_xlim(-0.5, 66)
    ax_path.set_ylim(-lane_y, lane_y)
    ax_path.set_xlabel("X — forward (m)", fontsize=9)
    ax_path.set_ylabel("Y — lateral (m)", fontsize=9)
    ax_path.tick_params(labelsize=8)
    ax_path.grid(True, alpha=0.3)
    ax_path.set_title("Sprint Track  (colour = thoracic velocity)",
                      fontsize=9, pad=2)

    writer = _MP4Writer(fig, out_path, fps)
    try:
        for n in range(0, n_frames, every):
            now_dot.set_data([time[n]], [vel[n]])
            info.set_text(f"Max V: {max_vel_mps:.2f} m/s\n"
                          f"Finish: {finish_time:.2f} s\n"
                          f"Now: {time[n]:.2f} s  ({vel[n]:.2f} m/s)")

            skel.set_frame(pts[n], h=0, v=2)
            ax_side.set_xlim(cam_x[n] - 1.5, cam_x[n] + 1.5)

            trail_rgba[:, 3] = seg_rgba[:, 3] * (np.arange(len(segs)) < n)
            trail.set_color(trail_rgba)
            path_dot.set_data([pts[n, MK_T12, 0]], [pts[n, MK_T12, 1]])
            if n >= peak_frame:
                peak_dot.set_data([pts[peak_frame, MK_T12, 0]],
                                  [pts[peak_frame, MK_T12, 1]])

            writer.add()
    finally:
        writer.close()
    return str(out_path)


def render_trial(pid, fps=None, every=1, dpi=100):
    """Convenience wrapper: participant ID in, MP4 path out."""
    matches = sorted(C3D_DIR.glob(f"{pid}*.c3d"))
    if not matches:
        raise SystemExit(f"No C3D file found for '{pid}'")
    out = cohort_dir(MP4_OUT, "Trial Runs") / f"{pid}_run.mp4"
    return Path(render_video(str(matches[0]), str(out), fps=fps, every=every,
                             dpi=dpi))


# ── Fast group against slow group, top speed ────────────────────────────────

def _analysis_cycle(pid, n_points=101):
    """One mean top-speed cycle of 64-marker positions for a participant.

    Uses the ANALYSIS cleaning, not the pipeline above, because this video is
    the animated form of pc1_anatomy.png and has to match it. Scaled to 1.75 m
    for drawing so averaging different-sized athletes does not blur the mean.
    """
    t = clean_for_angles(pid)
    windows, _ = find_top_speed_strides(t["mk"], t["peak_frame"])
    if not windows:
        return None, None

    raw  = t["raw64"] * height_scale_for(pid, 1.75)
    grid = np.linspace(0, 1, n_points)
    cycles = []
    for a, b in windows:
        seg = raw[a:b + 1]
        src = np.linspace(0, 1, len(seg))
        cycles.append(np.stack([[np.interp(grid, src, seg[:, m, ax])
                                 for ax in range(3)]
                                for m in range(seg.shape[1])]).transpose(2, 0, 1))
    return np.mean(cycles, axis=0), t["peak_vel_ms"]


def render_representative(n=3, out_path=None, fps=50, loops=4, dpi=110):
    """The `n` fastest athletes averaged into one skeleton, against the `n` slowest.

    Not one athlete each - the mean of `n`, so the comparison is of a
    representative fast sprinter against a representative slow one rather than
    two individuals who might each be unusual.

    The cycle is 101 samples of a stride that really lasts about 0.45 s, so at
    50 fps this plays at roughly a quarter speed and each frame advances the
    body by 1 % of the cycle. The old 12 fps showed the same 1 % steps four
    times more slowly, which is what made it stutter.
    """
    out_path = Path(out_path or MP4_OUT / f"representative_top{n}_vs_bottom{n}.mp4")

    # 1. Rank by real (unscaled) peak velocity.
    ranked = []
    for pid in cohort_pids():
        cyc, vel = _analysis_cycle(pid)
        if cyc is not None:
            ranked.append((vel, pid, cyc))
    ranked.sort(key=lambda r: r[0])

    slow, fast = ranked[:n], ranked[-n:]

    # 2. Average each group into one representative skeleton.
    groups = []
    for grp, colour, role in ((fast, COL_FAST, "faster"), (slow, COL_SLOW, "slower")):
        groups.append(dict(cyc=np.mean([c for _v, _p, c in grp], axis=0),
                           pids=[p for _v, p, _c in grp],
                           vel=np.mean([v for v, _p, _c in grp]),
                           colour=colour, role=role))

    # 3. One shared floor so neither skeleton floats.
    foot = [R_HEEL, L_HEEL, MK_R_TOE, MK_L_TOE]
    floor = min(float(g["cyc"][:, foot, 2].min()) for g in groups)

    # 4. Pre-centre every frame: pelvis over the origin, feet on the floor.
    #    Doing it here rather than inside the loop keeps the per-frame work to a
    #    set_data call.
    for g in groups:
        cyc = g["cyc"].copy()
        c = cyc[:, MK_PELVIS_ALL, :].mean(axis=1)         # (points, 3)
        c[:, 2] = floor                                   # keep height, drop to floor
        g["draw"] = cyc - c[:, None, :]

    # 100, not 101: sample 100 is the same instant as sample 0 of the next
    # cycle, so rendering it would show one frame twice at every loop join.
    n_pts = groups[0]["draw"].shape[0] - 1

    # Fixed axes rectangle and fixed text positions. Constrained layout re-solves
    # on every draw, so a title that changes width - "7 %" to "100 %" - nudged
    # the axes and made the whole frame shiver.
    fig = _agg_figure((7.6, 7.4), dpi)
    ax  = fig.add_axes([0.10, 0.07, 0.86, 0.79])
    f, s = groups[0], groups[1]
    fig.text(0.5, 0.975, "Representative sprinters — top speed",
             ha="center", va="top", fontsize=12.5, fontweight="bold")
    fig.text(0.5, 0.935,
             f"blue: {', '.join(f['pids'])}  ({f['vel']:.2f} m/s avg)      "
             f"red: {', '.join(s['pids'])}  ({s['vel']:.2f} m/s avg)",
             ha="center", va="top", fontsize=10)

    skels = [_Skeleton(ax, g["colour"], g["colour"], g["colour"], lw=2.0,
                       left_alpha=LEFT_ALPHA) for g in groups]
    phase = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=11,
                    fontweight="bold", ha="left", va="top")
    ax.axhline(0.0, color="#8a6d3b", lw=1.4, alpha=0.6, zorder=0)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.15, 2.05)
    ax.set_aspect("equal"); ax.grid(alpha=0.22)
    ax.set_xlabel("fore-aft (m)"); ax.set_ylabel("vertical (m)")

    writer = _MP4Writer(fig, out_path, fps)
    try:
        for _ in range(loops):
            for i in range(n_pts):
                for skel, g in zip(skels, groups):
                    skel.set_frame(g["draw"][i], h=0, v=2)
                phase.set_text(f"{i} % of the stride cycle")
                writer.add()
    finally:
        writer.close()
    return out_path


# ── Fast group against slow group, blocks through N steps ───────────────────

def _accel_sequence(pid, n_steps=10, per_step=30):
    """One athlete from the blocks through `n_steps`, on a step-anchored clock.

    Top speed could be averaged on a single stride cycle because every stride is
    a repeat of the last. Acceleration is a PROGRESSION - step 1 is not step 10 -
    so the steps are used as anchors instead: each step is resampled to
    `per_step` points, which guarantees that "step 5" is genuinely step 5 for
    every athlete however long they took to get there.

    Shape and position are separated on purpose:
      * shape    is scaled to 1.75 m so athletes of different sizes average into
                 a clean skeleton rather than a blur,
      * position is the TRUE, unscaled pelvis path, so the distance covered is
                 honest and not stretched by the rescale.

    Returns (shape, pelvis, seconds) or None.
    """
    t = clean_for_angles(pid)
    windows, _ = find_first_steps(t["mk"], n_steps=n_steps)
    if len(windows) < n_steps:
        return None

    raw   = t["raw64"]
    pelv  = raw[:, MK_PELVIS_ALL, :].mean(axis=1)          # true, unscaled
    hs    = height_scale_for(pid, 1.75)
    grid  = np.linspace(0, 1, per_step, endpoint=False)

    shapes, positions, secs = [], [], []
    for a, b in windows:
        seg_shape = (raw[a:b + 1] - pelv[a:b + 1, None, :]) * hs   # about the pelvis
        seg_pelv  = pelv[a:b + 1]
        src = np.linspace(0, 1, len(seg_shape))
        for u in grid:
            shapes.append(np.stack([[np.interp(u, src, seg_shape[:, m, ax])
                                     for ax in range(3)]
                                    for m in range(seg_shape.shape[1])]))
            positions.append(np.array([np.interp(u, src, seg_pelv[:, ax])
                                       for ax in range(3)]))
            secs.append((a + u * (b - a)) / t["fs"])
    return np.array(shapes), np.array(positions), np.array(secs)


def render_representative_accel(n=3, n_steps=10, per_step=30, out_path=None,
                                fps=50, dpi=110):
    """Blocks through `n_steps`, fast group against slow group, two panels.

    LEFT   pelvis-centred, so every difference is posture - the same view as the
           top-speed comparison.
    RIGHT  ground-referenced, so the step lengths and the gap that opens up are
           visible. The camera follows the pair.

    Both panels run on the same step-anchored clock. That is what makes the left
    panel meaningful, but it means the two groups are NOT at the same instant in
    time - the slower group takes longer to reach any given step. Elapsed time is
    printed for each group so that is never hidden.

    `per_step` is 30 rather than 20: each step is a separate resampling, so the
    number of samples per step IS the frame rate of the movement. 20 of them
    across a 0.2 s step is a coarse picture of a fast limb.
    """
    out_path = Path(out_path or MP4_OUT / f"representative_accel_top{n}_vs_bottom{n}.mp4")

    # 1. Rank by real peak velocity, using the same six subjects as top speed.
    ranked = sorted((clean_for_angles(pid)["peak_vel_ms"], pid)
                    for pid in cohort_pids())
    picks = {"fast": [p for _v, p in ranked[-n:]], "slow": [p for _v, p in ranked[:n]]}

    # 2. Average each group, shape and position separately.
    groups = []
    for key, colour, role in (("fast", COL_FAST, "faster"), ("slow", COL_SLOW, "slower")):
        seqs = [_accel_sequence(p, n_steps, per_step) for p in picks[key]]
        seqs = [s for s in seqs if s is not None]
        if not seqs:
            raise SystemExit(f"no usable {key} sequences")
        groups.append(dict(
            shape=np.mean([s[0] for s in seqs], axis=0),
            pelv =np.mean([s[1] for s in seqs], axis=0),
            secs =np.mean([s[2] for s in seqs], axis=0),
            pids=picks[key], colour=colour, role=role))

    n_pts = groups[0]["shape"].shape[0]
    # 3. One floor for both panels, from the feet across the whole sequence.
    foot = [R_HEEL, L_HEEL, MK_R_TOE, MK_L_TOE]
    floor = min(float((g["shape"] + g["pelv"][:, None, :])[:, foot, 2].min())
                for g in groups)

    # Both panels use an equal aspect, so their widths must be in the same ratio
    # as their x-spans or the ground panel renders as a small wide strip. Fixed
    # rectangles, not constrained layout: the header text changes every frame.
    HALF_P, HALF_G = 1.3, 1.9
    fig = _agg_figure((14.5, 6.6), dpi)
    LEFT, GAP, RIGHT = 0.045, 0.055, 0.015          # margins in figure fractions
    W = (1 - LEFT - GAP - RIGHT) * np.array([HALF_P, HALF_G]) / (HALF_P + HALF_G)
    ax_p = fig.add_axes([LEFT,                  0.09, W[0], 0.74])
    ax_g = fig.add_axes([LEFT + W[0] + GAP,     0.09, W[1], 0.74])

    f, s = groups[0], groups[1]
    fig.text(0.5, 0.985, f"Blocks to step {n_steps}", ha="center", va="top",
             fontsize=13, fontweight="bold")
    header = fig.text(0.5, 0.945, "", ha="center", va="top", fontsize=10.5)

    skel_p = [_Skeleton(ax_p, g["colour"], g["colour"], g["colour"], lw=2.0,
                        left_alpha=LEFT_ALPHA) for g in groups]
    skel_g = [_Skeleton(ax_g, g["colour"], g["colour"], g["colour"], lw=2.0,
                        left_alpha=LEFT_ALPHA) for g in groups]
    for ax in (ax_p, ax_g):
        ax.axhline(0.0, color="#8a6d3b", lw=1.4, alpha=0.6, zorder=0)
        ax.set_aspect("equal"); ax.grid(alpha=0.22)
        ax.set_xlabel("fore-aft (m)")
        ax.set_ylim(-0.15, 2.05)
    ax_p.set_xlim(-HALF_P, HALF_P)
    ax_p.set_ylabel("vertical (m)")
    ax_p.set_title("Posture — pelvis aligned", fontsize=11, fontweight="bold")
    ax_g.set_title("Ground — true distance from the blocks",
                   fontsize=11, fontweight="bold")

    # The ground camera follows the midpoint of the two groups. Averaged pelvis
    # tracks still carry the within-step surge, so the same smoothing the trial
    # renderer uses is applied here - otherwise the ground grid pulses once per
    # step underneath skeletons that are themselves moving evenly.
    mid_raw = np.mean([g["pelv"][:, 0] for g in groups], axis=0)
    mid_cam = _smooth_track(mid_raw, per_step, win_s=1.0)

    writer = _MP4Writer(fig, out_path, fps)
    try:
        for i in range(n_pts):
            for sp_art, sg_art, g in zip(skel_p, skel_g, groups):
                # LEFT - pelvis at the origin, posture only.
                sp = g["shape"][i].copy()
                sp[:, 2] += g["pelv"][i, 2] - floor
                sp_art.set_frame(sp, h=0, v=2)

                # RIGHT - true pelvis position, so distance is real.
                gp = g["shape"][i] + g["pelv"][i]
                gp[:, 2] -= floor
                sg_art.set_frame(gp, h=0, v=2)

            ax_g.set_xlim(mid_cam[i] - HALF_G, mid_cam[i] + HALF_G)
            header.set_text(
                f"step {i // per_step + 1} of {n_steps}          "
                f"blue {', '.join(f['pids'])}:  {f['pelv'][i,0]:.1f} m at {f['secs'][i]:.2f} s"
                f"      red {', '.join(s['pids'])}:  {s['pelv'][i,0]:.1f} m at {s['secs'][i]:.2f} s")
            writer.add()
    finally:
        writer.close()
    return out_path


# ── Fast group against slow group, the WHOLE run ────────────────────────────

def common_step_count(pids, cap=60):
    """The largest number of steps every athlete in `pids` actually has.

    The step-anchored clock only means anything if "step 20" exists for
    everybody, so the sequence has to stop at the shortest athlete. Over a full
    62 m these six run 29-38 steps, mostly because a slower athlete takes more
    of them to cover the same ground.
    """
    counts = []
    for pid in pids:
        t = clean_for_angles(pid)
        steps, _ = find_first_steps(t["mk"], n_steps=cap)
        counts.append(len(steps))
    return int(min(counts))


def render_representative_full(n=3, n_steps=None, per_step=20, out_path=None,
                               fps=50, dpi=110, track_m=62.5, body_scale=4.0):
    """The whole run, fast group against slow group, two panels.

    The acceleration video stops at step 10, about 16 m in. This one runs the
    trial out to the last step every athlete has — around 29 steps and 60 m — so
    it answers a different question: not "how do they leave the blocks" but
    "where does the gap actually open up".

    LEFT   pelvis-aligned posture, exactly the panel in the acceleration video.
    RIGHT  the ENTIRE track in one fixed view, blocks to finish, no camera
           following anybody. Both groups are on it at once, so the distance
           between them IS the result, and it is printed as a running gap.

    `body_scale` is the one compromise and it is on the panel in words. 62 m of
    track against a 2 m athlete is a 31:1 aspect: drawn true to scale the
    runners come out about forty pixels tall and you cannot see what they are
    doing. Each body is therefore drawn `body_scale` times life size ABOUT ITS
    OWN GROUND CONTACT POINT, which leaves every position — and therefore every
    distance and the gap — exactly where it really is. `body_scale=1.0` gives
    the undistorted view. The right panel's vertical axis is unlabelled because
    of it; vertical measurements belong to the left panel.

    The clock is step-anchored, like the acceleration video and for the same
    reason: it guarantees the three athletes in a group are at the same point of
    the same step, so their average is a body rather than a blur. It also means
    the two groups are NOT at the same instant in time — the slower group takes
    longer to reach any given step — so elapsed seconds are printed for each.
    """
    # 1. Same six athletes as every other group comparison.
    ranked = sorted((clean_for_angles(pid)["peak_vel_ms"], pid)
                    for pid in cohort_pids())
    picks = {"fast": [p for _v, p in ranked[-n:]], "slow": [p for _v, p in ranked[:n]]}

    if n_steps is None:
        n_steps = common_step_count(picks["fast"] + picks["slow"])
    out_path = Path(out_path or
                    MP4_OUT / f"representative_full_top{n}_vs_bottom{n}.mp4")

    # 2. Average each group, shape and position kept separate.
    groups = []
    for key, colour in (("fast", COL_FAST), ("slow", COL_SLOW)):
        seqs = [_accel_sequence(p, n_steps, per_step) for p in picks[key]]
        seqs = [q for q in seqs if q is not None]
        if not seqs:
            raise SystemExit(f"no usable {key} sequences")
        groups.append(dict(
            shape=np.mean([q[0] for q in seqs], axis=0),
            pelv =np.mean([q[1] for q in seqs], axis=0),
            secs =np.mean([q[2] for q in seqs], axis=0),
            pids=picks[key], colour=colour))

    n_pts = groups[0]["shape"].shape[0]
    foot  = [R_HEEL, L_HEEL, MK_R_TOE, MK_L_TOE]
    floor = min(float((g["shape"] + g["pelv"][:, None, :])[:, foot, 2].min())
                for g in groups)

    # 3. Layout, in inches first so both panels can be true to their own aspect.
    #    Fixed rectangles again: the header text changes every frame, and
    #    constrained layout would re-solve and shift the axes under it.
    FW, FH  = 19.2, 5.4
    HALF_P  = 1.3
    Y0, Y1  = -0.15, 2.05
    X0, X1  = -1.5, track_m + 1.5
    G0, G1  = -0.25, 2.15 * body_scale          # right panel, magnified bodies

    fig = _agg_figure((FW, FH), dpi)

    left_w_in  = 4.3
    left_h_in  = left_w_in * (Y1 - Y0) / (2 * HALF_P)
    right_w_in = 13.0
    right_h_in = right_w_in * (G1 - G0) / (X1 - X0)
    bottom_in  = 0.55

    ax_p = fig.add_axes([0.030, bottom_in / FH, left_w_in / FW, left_h_in / FH])
    ax_g = fig.add_axes([0.285,
                         (bottom_in + (left_h_in - right_h_in) / 2) / FH,
                         right_w_in / FW, right_h_in / FH])

    f, s_ = groups[0], groups[1]
    fig.text(0.5, 0.985, f"The whole run — blocks to step {n_steps}",
             ha="center", va="top", fontsize=13, fontweight="bold")
    header = fig.text(0.5, 0.938, "", ha="center", va="top", fontsize=10.5)

    skel_p = [_Skeleton(ax_p, g["colour"], g["colour"], g["colour"], lw=2.0,
                        left_alpha=LEFT_ALPHA) for g in groups]
    skel_g = [_Skeleton(ax_g, g["colour"], g["colour"], g["colour"], lw=1.5,
                        left_alpha=LEFT_ALPHA) for g in groups]
    # At this scale the trail is what carries the story: two lines pulling apart.
    trails = [ax_g.plot([], [], color=g["colour"], lw=1.5, alpha=0.5,
                        solid_capstyle="round")[0] for g in groups]

    for ax in (ax_p, ax_g):
        ax.axhline(0.0, color="#8a6d3b", lw=1.4, alpha=0.6, zorder=0)
        ax.set_aspect("equal")
    ax_p.set_xlim(-HALF_P, HALF_P); ax_p.set_ylim(Y0, Y1)
    ax_p.grid(alpha=0.22)
    ax_p.set_xlabel("fore-aft (m)")
    ax_p.set_ylabel("vertical (m)")
    ax_p.set_title("Posture — pelvis aligned", fontsize=11, fontweight="bold")

    ax_g.set_xlim(X0, X1); ax_g.set_ylim(G0, G1)
    ax_g.set_xticks(np.arange(0, track_m + 1, 10))
    ax_g.set_yticks([])                       # bodies are magnified: no vertical scale
    ax_g.grid(axis="x", alpha=0.3)
    ax_g.tick_params(labelsize=9)
    ax_g.set_xlabel("distance from the blocks (m)")
    ax_g.axvline(0.0, color="#888888", lw=1, ls="--")
    ax_g.axvline(track_m, color="k", lw=1.6)
    ax_g.text(track_m - 0.6, G1, f"{track_m:g} m", fontsize=9,
              fontweight="bold", va="top", ha="right")
    scale_note = ("true scale" if body_scale == 1
                  else f"positions true, bodies drawn {body_scale:g}x for visibility")
    ax_g.set_title(f"Ground — the entire {track_m:g} m, fixed camera "
                   f"({scale_note})", fontsize=11, fontweight="bold")

    writer = _MP4Writer(fig, out_path, fps)
    try:
        for i in range(n_pts):
            for sp_art, sg_art, trail, g in zip(skel_p, skel_g, trails, groups):
                # LEFT — pelvis at the origin, posture only.
                sp = g["shape"][i].copy()
                sp[:, 2] += g["pelv"][i, 2] - floor
                sp_art.set_frame(sp, h=0, v=2)

                # RIGHT — true pelvis position on a fixed track, body magnified
                # about its own ground point so the feet stay on the floor and
                # the fore-aft position stays exactly where it belongs.
                gp = g["shape"][i] + g["pelv"][i]
                gp[:, 2] -= floor
                gp[:, 0] = g["pelv"][i, 0] + (gp[:, 0] - g["pelv"][i, 0]) * body_scale
                gp[:, 2] *= body_scale
                sg_art.set_frame(gp, h=0, v=2)
                trail.set_data(g["pelv"][: i + 1, 0],
                               (g["pelv"][: i + 1, 2] - floor) * body_scale)

            gap = f["pelv"][i, 0] - s_["pelv"][i, 0]
            header.set_text(
                f"step {i // per_step + 1} of {n_steps}          "
                f"blue {', '.join(f['pids'])}:  {f['pelv'][i,0]:.1f} m at {f['secs'][i]:.2f} s"
                f"      red {', '.join(s_['pids'])}:  {s_['pelv'][i,0]:.1f} m at {s_['secs'][i]:.2f} s"
                f"      gap {gap:+.1f} m")
            writer.add()
    finally:
        writer.close()
    return out_path


# ==========================================================================
# 9 · Run everything, for everyone
# ==========================================================================
#
# Sections 1-6 are where the testing happens, in sprint_pipeline.ipynb. So this
# does not pick a representative athlete or a demonstration trial: it renders
# the whole cohort, every time.


def build_analysis(n_points=101, accel_steps=3, verbose=True):
    """One pass over the cohort producing everything the FIGURES need.

    Returns a dict with the top-speed and acceleration angle cycles, the
    matching 64-marker cycles for the skeleton figures, both fPCA fits, and
    peak velocity and stature per athlete.

    Angle curves are smoothed before the fPCA; the marker cycles are not,
    because they are drawn rather than decomposed.
    """
    cohort, cohort_raw, accel, accel_raw = {}, {}, {}, {}
    peak_vel, body_height, spread_top, spread_acc = {}, {}, [], []

    for fpath in sorted(C3D_DIR.glob("*.c3d")):
        pid = fpath.stem.split("-")[0].strip()
        if pid in EXCLUDED_PIDS:
            continue
        try:
            tr = clean_for_angles(fpath)
            hs = height_scale_for(pid, 1.75)            # drawing only

            cy, _n, sp = stride_cycle_angles(tr["raw64"], tr["mk"],
                                             tr["peak_frame"], n_points)
            if cy is None:
                continue
            cohort[pid]     = cy
            cohort_raw[pid] = stride_cycle_raw64(tr["raw64"], tr["mk"],
                                                 tr["peak_frame"], hscale=hs,
                                                 n_points=n_points)
            spread_top.append(sp)

            steps, _ = find_first_steps(tr["mk"], n_steps=accel_steps)
            if len(steps) == accel_steps:
                ac, _na, sa = stride_cycle_angles(tr["raw64"], tr["mk"],
                                                  tr["peak_frame"], n_points,
                                                  windows=steps)
                if ac is not None:
                    accel[pid]     = ac
                    accel_raw[pid] = stride_cycle_raw64(tr["raw64"], tr["mk"],
                                                        tr["peak_frame"],
                                                        hscale=hs,
                                                        n_points=n_points,
                                                        windows=steps)
                    spread_acc.append(sa)

            peak_vel[pid]    = tr["peak_vel_ms"]
            body_height[pid] = PARTICIPANT_ANTHRO[pid]["body_height"]
        except Exception as e:
            print(f"  skipped {pid}: {e}")

    smooth_top = dict(zip(sorted(cohort),
                          smooth_angle_curves(np.stack([cohort[p] for p in sorted(cohort)]),
                                              penalty=1.0)))
    smooth_acc = dict(zip(sorted(accel),
                          smooth_angle_curves(np.stack([accel[p] for p in sorted(accel)]),
                                              penalty=1.0)))

    out = {"cohort": cohort, "cohort_raw": cohort_raw,
           "accel": accel, "accel_raw": accel_raw,
           "smooth_top": smooth_top, "smooth_acc": smooth_acc,
           "fpca_top": run_angle_fpca(smooth_top, n_components=6),
           "fpca_acc": run_angle_fpca(smooth_acc, n_components=6),
           "peak_vel": peak_vel, "body_height": body_height,
           "phase_spread_top": float(np.mean(spread_top)) if spread_top else np.nan,
           "phase_spread_acc": float(np.mean(spread_acc)) if spread_acc else np.nan}

    if verbose:
        print(f"  {len(cohort)} athletes at top speed, {len(accel)} with "
              f"{accel_steps} clean steps out of the blocks")
        print(f"  phase spread: {out['phase_spread_top']:.0f} % top speed, "
              f"{out['phase_spread_acc']:.0f} % acceleration")
    return out


def model_specs(feat):
    """The candidate models, simplest first, as `model_ladder` wants them.

    Hip extension is gone from this ladder. It was a pre-specified hypothesis
    about a quantity nobody asked for, and both the feature and its figure have
    been removed; knee range of motion carries the leg signal here.
    """
    from sklearn.linear_model import RidgeCV

    ridge = lambda: RidgeCV(alphas=np.logspace(-3, 4, 60))
    X, y, height = feat["X"], feat["y"], feat["height"]
    pids = list(y.index)
    n    = len(pids)
    h    = height.values

    knee  = X[["knee_hi_ROM", "knee_lo_ROM"]].values
    combo = np.column_stack([X.knee_hi_ROM, X.knee_lo_ROM, h])

    specs = [
        ("intercept only",           np.zeros((n, 1)), None,  None),
        ("height only",              h[:, None],       None,  None),
        ("knee ROM (hi + lo)",       knee,             None,  None),
        ("knee ROM + height",        combo,            None,  None),
        ("all scalars, ridge",       X.values,         ridge, None),
        ("fPC 1-6 R/L, ridge",       None, ridge, fpca_fold_transform(feat["cycles"], pids, 6)),
        ("fPC 1-6 HI/LO, ridge",     None, ridge, fpca_fold_transform(feat["hilo"],   pids, 6)),
    ]
    return specs, combo, pids


def run_all_outputs(n_group=3, n_steps=10, every=1, dpi=100, n_perm=5000,
                    trials=True, verbose=True):
    """Every figure and every MP4, for every athlete. The whole output stage.

    `trials=False` skips the per-athlete videos, which are the slow part.
    `every=2` halves the frames in those videos for a quick look — the frame
    rate drops with it, so the result is real time but coarser.

    Returns a dict of everything written, keyed by kind.
    """
    written = {"figures": [], "mp4s": []}
    pids_all = cohort_pids()

    def _say(msg):
        if verbose:
            print(msg, flush=True)

    # ── A. The analysis the figures are drawn from ──────────────────────────
    _say(f"Building the cohort ({len(pids_all)} athletes)…")
    an   = build_analysis(verbose=verbose)
    feat = build_feature_table(verbose=verbose)

    # ── B. Section 7 — figures ──────────────────────────────────────────────
    # Every plot_* function leaves its figure open, so that a notebook shows it
    # once. Nothing is watching here and 38 open figures trips matplotlib's own
    # warning, so each one is closed as soon as it is on disk.
    _say("\nFigures →")

    def _save(fn, *a, **kw):
        out = fn(*a, **kw)
        plt.close("all")
        written["figures"].append(out)
        return out

    pv = an["peak_vel"]
    _save(plot_fast_vs_slow, an["smooth_top"], pv, n=5)
    _save(plot_pc_loading_ranking, an["fpca_top"], pc=1)
    _save(plot_pc_anatomy, an["cohort_raw"], an["fpca_top"], pv, pc=1, k=2.0)
    _save(plot_pc_anatomy, an["accel_raw"], an["fpca_acc"], pv, pc=1, k=2.0,
          title_extra=" — acceleration, first 3 steps",
          fname="pc1_anatomy_acceleration.png")
    _save(plot_velocity_fit, feat)
    _save(plot_limb_asymmetry, feat)

    # Block exit, one panel row per athlete — the phase is athlete-specific, so
    # every athlete gets their own.
    for pid in pids_all:
        try:
            _save(plot_block_exit_steps, pid)
        except Exception as e:
            print(f"  skipped block-exit {pid}: {e}")

    # The model figures need the ladder run first.
    specs, combo, pids = model_specs(feat)
    y = feat["y"].values
    ladder, _preds = model_ladder(specs, y, verbose=verbose)
    null_r2 = float(ladder.loc[ladder.model == "intercept only", "LOO_R2"].iloc[0])
    pred = loo_predict(combo, y)
    obs_r2, p_perm, null = permutation_test(combo, y, n_perm=n_perm, seed=1)
    _say(f"  final model LOO R2 = {obs_r2:+.3f}, permutation p = {p_perm:.4f}")
    _save(plot_model_comparison, ladder, null_r2=null_r2)
    _save(plot_diagnostics, y, pred, combo, pids, null, obs_r2)
    for f in written["figures"]:
        _say(f"  {f.name}")

    # ── C. Section 8 — MP4s, everyone ───────────────────────────────────────
    _say(f"\nMP4s → {MP4_OUT}")
    if trials:
        for i, pid in enumerate(pids_all, 1):
            try:
                out = render_trial(pid, every=every, dpi=dpi)
                written["mp4s"].append(out)
                _say(f"  [{i:2d}/{len(pids_all)}] {out.name:<28} "
                     f"{out.stat().st_size / 1e6:5.1f} MB")
            except Exception as e:
                print(f"  [{i:2d}/{len(pids_all)}] skipped {pid}: {e}")

    for label, fn in (("top speed",    lambda: render_representative(n=n_group)),
                      ("acceleration", lambda: render_representative_accel(
                          n=n_group, n_steps=n_steps)),
                      ("whole run",    lambda: render_representative_full(
                          n=n_group))):
        try:
            out = fn()
            written["mp4s"].append(out)
            _say(f"  group, {label:<13} {out.name:<40} "
                 f"{out.stat().st_size / 1e6:5.1f} MB")
        except Exception as e:
            print(f"  group {label} failed: {e}")

    _say(f"\n{len(written['figures'])} figures in {FIG_OUT}")
    _say(f"{len(written['mp4s'])} MP4s in {MP4_OUT}")
    return written


if __name__ == "__main__":
    # `python sprint_outputs.py --no-trials` for the figures and the two group
    # animations only; `--every 2` for coarser, quicker per-athlete videos.
    argv = sys.argv[1:]
    kwargs = {}
    if "--no-trials" in argv:
        kwargs["trials"] = False
    if "--every" in argv:
        kwargs["every"] = int(argv[argv.index("--every") + 1])
    run_all_outputs(**kwargs)
