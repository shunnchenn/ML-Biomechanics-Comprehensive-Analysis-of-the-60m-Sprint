"""
_accel_fpca_step_ranges.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

Standalone script to:
  1) re-process every .c3d trial
  2) detect strides + step events for the acceleration phase
  3) build per-participant time-normalised mean strides for Steps 3-8 and Steps 9-16
  4) run fPCA on each step range
  5) render fast / median / slow skeleton overlay figures
     × 3 cohorts (Males / Females / All)
     × 2 step ranges (3-8, 9-16)
     = 6 figures total

Reuses constants + helpers from notebooks/02_Kinematics_PCA.ipynb cells 1, 2, 3
via exec (no duplication of the canonical pipeline code).

Output: outputs/figures/accel_fpca_steps_{3-8,9-16}_{M,F,all}.png
        outputs/figures/accel_fpca_steps_{3-8,9-16}_scree.png
        outputs/data/accel_fpca_steps_{3-8,9-16}_scores.csv

Local-only script (gitignored). The new cell will be added to
02_Kinematics_PCA.ipynb separately.
"""
from __future__ import annotations
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import find_peaks

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

NB_PATH = Path("notebooks/02_Kinematics_PCA.ipynb")

# ────────────────────────────────────────────────────────────────────────────
# 1) Load constants + helpers from notebook cells 1, 2, 3
# ────────────────────────────────────────────────────────────────────────────
print("[1/6] Loading notebook helpers...")
with open(NB_PATH) as f:
    _nb = json.load(f)
ns: dict = {"__name__": "__nb_helpers__"}
for cell_idx in (1, 2, 3, 4):
    src = "".join(_nb["cells"][cell_idx]["source"])
    # Strip Jupyter magics (cell 1 uses %run via get_ipython) — we only need
    # the canonical helpers from cell 3, not the (FINAL) Sprint_Ensemble_Viz overrides
    src = "\n".join(
        line for line in src.split("\n")
        if "get_ipython" not in line
    )
    exec(compile(src, f"<cell {cell_idx}>", "exec"), ns)
globals().update(ns)
print(f"      → loaded {sum(1 for k in ns if not k.startswith('_'))} symbols")

# Need a few extra axis constants from cell 15
_AP, _ML, _VERT = 0, 1, 2

# ────────────────────────────────────────────────────────────────────────────
# 2) Load velocity traces (peak velocity per participant)
# ────────────────────────────────────────────────────────────────────────────
print("[2/6] Loading velocity traces...")
with open(DATA_DIR / "velocity_traces.pkl", "rb") as f:
    velocity_traces = pickle.load(f)
print(f"      → {len(velocity_traces)} participants")


# ────────────────────────────────────────────────────────────────────────────
# 3) Helpers borrowed verbatim from cell 15
# ────────────────────────────────────────────────────────────────────────────
def detect_block_clearance_events(raw64, fs):
    """Block-clearance event detection with FIX for Front Foot Clearance.

    Previous bug: Front Foot Clearance searched from `Rear Ankle Cross` onwards
    for the first frame the front HEEL rose ≥ 5 mm above its initial position.
    By the time the rear ankle has crossed the front knee, the front foot has
    already begun to roll forward and the heel is often already >5 mm above
    the floor, so the loop returned `i = Rear Ankle Cross` immediately.
    That made the two events identical in every kinogram column.

    New logic:
        1. Use the front TOE marker (not heel): the toe is the LAST contact
           point of the foot still in the block — that is the true clearance.
        2. Require a 25 mm rise from baseline (vs. the previous 5 mm) — well
           above noise / heel float.
        3. Enforce at least 2 frames of separation from Rear Ankle Cross.
    """
    n = raw64.shape[0]
    first_sec = min(n, int(fs * 1.5))
    r_heel_z = raw64[:first_sec, MK_R_HEEL, _VERT]
    l_heel_z = raw64[:first_sec, MK_L_HEEL, _VERT]
    r_heel_x = raw64[:first_sec, MK_R_HEEL, _AP]
    l_heel_x = raw64[:first_sec, MK_L_HEEL, _AP]
    if r_heel_x[0] <= l_heel_x[0]:
        rear_heel_z, front_heel_z = r_heel_z, l_heel_z
        rear_mk,     front_mk     = MK_R_HEEL, MK_L_HEEL
        front_toe_mk = MK_L_TOE
        front_knee_mks = MK_L_KNEE
    else:
        rear_heel_z, front_heel_z = l_heel_z, r_heel_z
        rear_mk,     front_mk     = MK_L_HEEL, MK_R_HEEL
        front_toe_mk = MK_R_TOE
        front_knee_mks = MK_R_KNEE
    ev = {}
    for i in range(1, len(rear_heel_z)):
        if rear_heel_z[i] > rear_heel_z[0] + 0.005:
            ev["Rear Foot Clearance"] = i; break
    ev.setdefault("Rear Foot Clearance", 2)
    front_knee_x = np.mean(raw64[:first_sec, front_knee_mks, _AP], axis=1)
    rear_heel_ap = raw64[:first_sec, rear_mk, _AP]
    for i in range(ev["Rear Foot Clearance"], len(rear_heel_ap)):
        if rear_heel_ap[i] >= front_knee_x[i]:
            ev["Rear Ankle Cross"] = i; break
    ev.setdefault("Rear Ankle Cross", ev["Rear Foot Clearance"] + 5)

    # FIXED: detect Front Foot Clearance from TOE marker, ≥25 mm rise,
    #        and at least 2 frames after Rear Ankle Cross
    front_toe_z = raw64[:first_sec, front_toe_mk, _VERT]
    toe_init = front_toe_z[0]
    min_search = ev["Rear Ankle Cross"] + 2
    for i in range(min_search, len(front_toe_z)):
        if front_toe_z[i] > toe_init + 0.025:
            ev["Front Foot Clearance"] = i; break
    ev.setdefault("Front Foot Clearance", ev["Rear Ankle Cross"] + 8)
    return ev, rear_mk, front_mk


# ── Anthropometric scaling: rescale every athlete to TARGET_HEIGHT_M ──
TARGET_HEIGHT_M = 1.75   # standardised standing height for all figures


def _apparent_height_m(raw64: np.ndarray) -> float:
    """Standing-height proxy in metres: max(head_z) − min(foot_z) over trial.

    Robust against noise — the max head height occurs once the athlete reaches
    upright posture; min foot height = floor. Difference ≈ standing height.
    Returns NaN if implausible (outside 0.5–2.5 m).
    """
    if raw64.shape[0] < 1:
        return float("nan")
    head_z = raw64[:, MK_HEAD, 2].mean(axis=1)
    foot_z = np.minimum.reduce([
        raw64[:, MK_R_HEEL, 2],
        raw64[:, MK_L_HEEL, 2],
        raw64[:, MK_R_TOE,  2],
        raw64[:, MK_L_TOE,  2],
    ])
    h = float(head_z.max() - foot_z.min())
    return h if 0.5 < h < 2.5 else float("nan")


def _height_scale(raw64: np.ndarray, target_m: float = TARGET_HEIGHT_M) -> float:
    """Per-participant scale factor so apparent standing height = `target_m`."""
    h = _apparent_height_m(raw64)
    if not np.isfinite(h) or h <= 0:
        return 1.0
    return target_m / h


def detect_accel_strides_extended(raw64, fs, max_steps=16, start_frame=0):
    """Return up to `max_steps` step touchdown frames (foot-alternating)."""
    r_heel_z = raw64[:, MK_R_HEEL, _VERT]
    l_heel_z = raw64[:, MK_L_HEEL, _VERT]
    r_contacts, _ = find_peaks(-r_heel_z, distance=15, prominence=0.01)
    l_contacts, _ = find_peaks(-l_heel_z, distance=15, prominence=0.01)
    r_contacts = r_contacts[r_contacts >= start_frame]
    l_contacts = l_contacts[l_contacts >= start_frame]
    all_contacts = sorted(list(r_contacts) + list(l_contacts))
    contacts_with_foot = [
        (c, "R" if r_heel_z[c] <= l_heel_z[c] else "L") for c in all_contacts
    ]
    cleaned = [contacts_with_foot[0]] if contacts_with_foot else []
    for c, f in contacts_with_foot[1:]:
        if c - cleaned[-1][0] > 8:
            cleaned.append((c, f))
    if len(cleaned) >= 2:
        alt = [cleaned[0]]
        for c, f in cleaned[1:]:
            if f != alt[-1][1]:
                alt.append((c, f))
            else:
                prev_c = alt[-1][0]
                if r_heel_z[c] + l_heel_z[c] < r_heel_z[prev_c] + l_heel_z[prev_c]:
                    alt[-1] = (c, f)
        cleaned = alt
    return cleaned[: max_steps + 1]


def stride_window(raw64, td_frame, next_td, target_len=TN_POINTS):
    """Return a (target_len, 64, 3) time-normalised stride window."""
    seg = raw64[td_frame:next_td]
    n_in = seg.shape[0]
    if n_in < 3:
        return None
    # piecewise linear interp from n_in → target_len per (marker, axis)
    src_x = np.linspace(0.0, 1.0, n_in)
    tgt_x = np.linspace(0.0, 1.0, target_len)
    out = np.zeros((target_len, 64, 3))
    for m in range(64):
        for a in range(3):
            out[:, m, a] = np.interp(tgt_x, src_x, seg[:, m, a])
    return out


def _detect_stance_events(raw64, td_frame, next_td, support_foot):
    """Detect (touchdown, mid-stance, toe-off) frames normalised to TN_POINTS-1.

    Biomechanical definitions:
    - Touchdown : first frame the foot touches the ground (window start, frame 0)
    - Toe-Off   : last frame the support-foot heel is in contact with the floor
                  (heel-z below baseline + 50 mm)
    - Mid-Stance: temporal midpoint of the detected stance phase — a robust
                  proxy for the instant COM passes over the support foot.  The
                  pelvis-marker zero-crossing definition is unreliable in this
                  dataset because the foot often lands close to the pelvis at
                  acceleration steps, putting the argmin at frame 0.
    """
    heel_mk = MK_R_HEEL if support_foot == "R" else MK_L_HEEL
    n_raw = next_td - td_frame
    if n_raw < 5:
        return 0, TN_POINTS // 8, TN_POINTS // 3

    heel_z = raw64[td_frame:next_td, heel_mk, 2]
    # Baseline = ground level estimated from the first 3 frames (foot on ground)
    z_baseline = heel_z[: min(3, n_raw)].mean()
    threshold  = z_baseline + 0.05   # 50 mm above baseline = clearly off the ground

    toe_rel = 0
    for i in range(1, n_raw):
        if heel_z[i] > threshold:
            toe_rel = max(1, i - 1)
            break
    if toe_rel == 0:
        toe_rel = int(n_raw * 0.40)

    # Clamp to a plausible stance range (20-55 % of the step cycle)
    toe_rel = max(int(n_raw * 0.20), min(int(n_raw * 0.55), toe_rel))

    # Mid-stance = midpoint of the detected stance phase
    mid_rel = toe_rel // 2

    scale = (TN_POINTS - 1) / max(1, n_raw - 1)
    return 0, int(round(mid_rel * scale)), int(round(toe_rel * scale))


# ────────────────────────────────────────────────────────────────────────────
# 4) Re-process each .c3d → step-range mean strides (R-touchdown only)
# ────────────────────────────────────────────────────────────────────────────
print("[3/6] Processing C3D trials...")
STEP_RANGES = {
    "steps_3-8":  (3, 8),     # inclusive step indices (1-based per kinogram naming)
    "steps_9-16": (9, 16),
}

# Per-range: dict pid -> (TN_POINTS, 64, 3) mean stride (R-touchdown only)
range_data:   dict[str, dict[str, np.ndarray]] = {k: {} for k in STEP_RANGES}
# Per-range: dict pid -> (td_norm, mid_norm, toe_norm) median across strides
range_events: dict[str, dict[str, tuple]]      = {k: {} for k in STEP_RANGES}

c3d_files = sorted(C3D_DIR.glob("*.c3d"))
for fpath in c3d_files:
    pid = fpath.stem.split("-")[0].strip()
    if pid in EXCLUDED_PIDS or pid not in velocity_traces:
        continue
    try:
        mk, raw64, fs = load_c3d(fpath)
        mk, raw64 = perform_pca_alignment(mk, raw64)
        start = detect_sprint_start(mk, fs)
        mk = mk[start:]; raw64 = raw64[start:]
        mk = shift_origin(mk)
        r_x = raw64[0, MK_R_HEEL, _AP]; l_x = raw64[0, MK_L_HEEL, _AP]
        orig = raw64[0, MK_R_HEEL if r_x <= l_x else MK_L_HEEL].copy()
        raw64 = raw64 - orig
        mk = trim_sprint(mk); raw64 = raw64[: mk.shape[0]]

        # Anthropometric normalisation — every participant rescaled so apparent
        # standing height = TARGET_HEIGHT_M (1.75 m).  Body-segment proportions
        # are preserved because the same uniform scalar multiplies every joint.
        hscale = _height_scale(raw64, TARGET_HEIGHT_M)
        raw64  = raw64 * hscale

        blk_ev, _, _ = detect_block_clearance_events(raw64, fs)
        sstart = blk_ev.get("Front Foot Clearance", 0) + int(fs * 0.05)
        contacts = detect_accel_strides_extended(raw64, fs, max_steps=16, start_frame=sstart)
        if len(contacts) < 4:
            continue

        # Filter to RIGHT-touchdown strides only so the support foot is consistent
        # across averaged strides and across all participants.  Mixing R-TD and
        # L-TD strides creates a symmetric mean where neither foot is clearly on
        # the ground at any frame, which makes Touchdown / Mid-Stance / Toe-Off
        # all look identical in the overlays.
        for range_name, (lo, hi) in STEP_RANGES.items():
            strides     = []
            events_list = []
            for step_i in range(lo - 1, min(hi, len(contacts) - 1)):
                td, foot = contacts[step_i]
                if foot != "R":
                    continue
                next_td, _ = contacts[step_i + 1]
                w = stride_window(raw64, td, next_td)
                if w is None:
                    continue
                events_list.append(_detect_stance_events(raw64, td, next_td, foot))
                strides.append(w)
            if strides:
                range_data[range_name][pid]   = np.mean(strides, axis=0)
                ev_arr = np.array(events_list)
                range_events[range_name][pid] = tuple(int(v) for v in np.median(ev_arr, axis=0))
    except Exception as e:
        print(f"    SKIP {pid}: {e}")
        continue

for k, d in range_data.items():
    print(f"      {k}: {len(d)} participants")


# ────────────────────────────────────────────────────────────────────────────
# 5) fPCA per step range + scree plot
# ────────────────────────────────────────────────────────────────────────────
print("[4/6] Running fPCA + scree plots...")
fpca_results = {}
for range_name, per_pid in range_data.items():
    pids = sorted(per_pid.keys())
    X = np.stack([per_pid[p].reshape(-1) for p in pids])     # (n_pid, 19392)
    y = np.array([velocity_traces[p]["peak_vel"] for p in pids])
    mu = X.mean(axis=0)
    Xc = X - mu

    n_comp = min(15, len(pids) - 1)
    p = PCA(n_components=n_comp, random_state=RANDOM_SEED)
    scores = p.fit_transform(Xc)
    # correlation with peak velocity per PC
    corrs = np.array([np.corrcoef(scores[:, k], y)[0, 1] for k in range(n_comp)])

    fpca_results[range_name] = dict(
        pids=pids, X=X, mu=mu, scores=scores, pca=p, y=y, corrs=corrs
    )

    # save scores CSV
    df = pd.DataFrame(scores, index=pids,
                      columns=[f"PC{k+1}" for k in range(n_comp)])
    df.insert(0, "peak_vel_ms", y)
    df.to_csv(DATA_DIR / f"accel_fpca_{range_name}_scores.csv")

    # scree
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    xs = range(1, n_comp + 1)
    ax1.bar(xs, p.explained_variance_ratio_ * 100, color="#e07b54", alpha=0.85)
    ax1.set(xlabel="PC", ylabel="Variance (%)",
            title=f"fPCA — {range_name} — Per-PC Variance")
    ax2.bar(xs, np.abs(corrs), color="#4c72b0", alpha=0.85)
    for k, c in enumerate(corrs):
        ax2.text(k + 1, abs(c) + 0.01, f"{c:+.2f}",
                 ha="center", fontsize=7,
                 color="#4c72b0" if c > 0 else "#c44e52")
    ax2.set(xlabel="PC", ylabel="|r| with peak velocity",
            title=f"fPCA — {range_name} — Correlation w/ peak velocity")
    ax2.set_ylim(0, max(0.6, np.abs(corrs).max() * 1.25))
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"accel_fpca_{range_name}_scree.png",
                dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"      {range_name}: top-3 |r|={np.abs(corrs)[:3].round(2)}  cumvar(3)={p.explained_variance_ratio_[:3].sum()*100:.1f}%")


# ────────────────────────────────────────────────────────────────────────────
# 6) Render fast / median / slow overlay figures per cohort
# ────────────────────────────────────────────────────────────────────────────
print("[5/6] Rendering overlay figures...")

# Bone polylines (re-use from sprint_animation.py)
sys.path.insert(0, str(SCRIPT_DIR))
from sprint_animation import AXIAL_IDX, RIGHT_IDX, LEFT_IDX

COLOR_SLOW = "#c44e52"   # red
COLOR_MED  = "#666666"   # grey
COLOR_FAST = "#3a76b3"   # blue


def _cohort_phase_points(range_name, pids):
    """Median (touchdown, mid-stance, toe-off) frame indices across the cohort.

    Uses per-participant medians stored in `range_events` so the phase points
    actually correspond to the biomechanical events for that cohort & range
    (instead of fixed % guesses).
    """
    avail = [range_events[range_name][p] for p in pids if p in range_events[range_name]]
    if not avail:
        return [("Touchdown", 0), ("Mid-Stance", 20), ("Toe-Off", 40)]
    arr = np.array(avail)
    td, mid, toe = (int(v) for v in np.median(arr, axis=0))
    # Guarantee strictly-increasing, spread-out frames
    mid = max(td + 5, mid)
    toe = max(mid + 5, toe)
    return [("Touchdown", td), ("Mid-Stance", mid), ("Toe-Off", toe)]


def _draw_overlay(ax, frame_pts, color, alpha, lw, plane="xz"):
    if plane == "xz":
        h, v = frame_pts[:, 0], frame_pts[:, 2]
    else:  # frontal: y horiz, z vert
        h, v = frame_pts[:, 1], frame_pts[:, 2]
    for poly in AXIAL_IDX + RIGHT_IDX + LEFT_IDX:
        ax.plot(h[poly], v[poly], color=color, lw=lw, alpha=alpha,
                solid_capstyle="round")


def _render_overlay_figure(range_name, cohort_label, pids_in_cohort, per_pid_strides):
    if len(pids_in_cohort) < 3:
        print(f"      skip {range_name} {cohort_label}: only {len(pids_in_cohort)} participants")
        return None
    # Rank by peak velocity within cohort
    ranked = sorted(pids_in_cohort, key=lambda p: velocity_traces[p]["peak_vel"])
    slow_pid = ranked[0]
    fast_pid = ranked[-1]
    med_pid  = ranked[len(ranked) // 2]
    v_slow = velocity_traces[slow_pid]["peak_vel"]
    v_med  = velocity_traces[med_pid]["peak_vel"]
    v_fast = velocity_traces[fast_pid]["peak_vel"]

    # Pull mean strides (TN_POINTS, 64, 3) for each (in metres, scale ×1000 for mm)
    s_slow = per_pid_strides[slow_pid] * SCALE_MM
    s_med  = per_pid_strides[med_pid]  * SCALE_MM
    s_fast = per_pid_strides[fast_pid] * SCALE_MM

    # Cohort-specific biomechanical phase frames
    phase_pts = _cohort_phase_points(range_name, pids_in_cohort)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5),
                             constrained_layout=True)
    cols = ["Touchdown", "Mid-Stance", "Toe-Off"]
    rows = ["Sagittal", "Frontal"]

    for row, (plane_name, plane_code) in enumerate(zip(rows, ["xz", "yz"])):
        for col, (phase_name, frame_idx) in enumerate(phase_pts):
            ax = axes[row, col]
            for stride, color, alpha, lw, label in [
                (s_slow, COLOR_SLOW, 0.75, 1.8, f"Slowest ({v_slow:.1f} m/s)"),
                (s_med,  COLOR_MED,  0.55, 1.2, f"Median ({v_med:.1f} m/s)"),
                (s_fast, COLOR_FAST, 0.75, 1.8, f"Fastest ({v_fast:.1f} m/s)"),
            ]:
                pts = stride[frame_idx]
                # Centre horizontally (X, Y) on pelvis so all 3 skeletons align,
                # but keep Z natural so legs reach the floor and head sits on top.
                centre = pts[MK_PELVIS].copy()
                centre[2] = 0.0
                pts = pts - centre
                # Drop the skeleton so the lowest foot point sits at z = 0 (the floor)
                z_min = pts[:, 2].min()
                pts[:, 2] = pts[:, 2] - z_min
                _draw_overlay(ax, pts, color, alpha, lw, plane=plane_code)

            # Ground line
            ax.axhline(0, color="#999999", lw=0.8, ls="-", alpha=0.6, zorder=0)

            if row == 0:
                ax.set_title(phase_name, fontsize=11, pad=6)
            if col == 0:
                ax.set_ylabel(plane_name, fontsize=11, fontweight="bold")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=8)
            # Full-body framing: floor → above head (mm)
            if plane_code == "xz":
                ax.set_xlim(-900, 1100); ax.set_ylim(-50, 2050)
            else:
                ax.set_xlim(-800, 800); ax.set_ylim(-50, 2050)

    fpca = fpca_results[range_name]
    top_pc = int(np.argmax(np.abs(fpca["corrs"]))) + 1
    top_r  = fpca["corrs"][top_pc - 1]
    top_ev = fpca["pca"].explained_variance_ratio_[top_pc - 1] * 100
    label_pretty = {"M": "Males", "F": "Females", "all": "All Participants"}[cohort_label]
    fig.suptitle(
        f"Acceleration fPCA — {range_name.replace('_', ' ').replace('steps', 'Steps')} "
        f"— {label_pretty}  (n = {len(pids_in_cohort)})\n"
        f"Strongest velocity-correlated component: PC{top_pc}  "
        f"({top_ev:.1f}% var, r = {top_r:+.2f} with peak velocity)\n"
        f"All skeletons scaled to {TARGET_HEIGHT_M:.2f} m (same anthropometrics)",
        fontsize=11, fontweight="bold")

    # Single legend at bottom
    handles = [
        plt.Line2D([], [], color=COLOR_SLOW, lw=2.5, label=f"Slowest ({v_slow:.1f} m/s)"),
        plt.Line2D([], [], color=COLOR_MED,  lw=2.0, label=f"Median ({v_med:.1f} m/s)"),
        plt.Line2D([], [], color=COLOR_FAST, lw=2.5, label=f"Fastest ({v_fast:.1f} m/s)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=10, frameon=False)

    out_path = FIG_DIR / f"accel_fpca_{range_name}_{cohort_label}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


# Render 6 figures: 2 step ranges × 3 cohorts
for range_name, per_pid in range_data.items():
    cohorts = {
        "M":   [p for p in per_pid if PARTICIPANT_SEX.get(p) == "M"],
        "F":   [p for p in per_pid if PARTICIPANT_SEX.get(p) == "F"],
        "all": list(per_pid.keys()),
    }
    for c_label, pids in cohorts.items():
        op = _render_overlay_figure(range_name, c_label, pids, per_pid)
        if op:
            print(f"      wrote {op.name}")


# ────────────────────────────────────────────────────────────────────────────
# 6b) PC selection (per range) + per-PC overlay + shape-mode figures
# ────────────────────────────────────────────────────────────────────────────
print("[5b] Selecting PCs by 95% kinematic variance + top-3 |r| with velocity …")
from scipy.stats import pearsonr

TARGET_VAR = 0.95   # cumulative kinematic variance for the "variance" selection


def _select_pcs(pca: PCA, scores: np.ndarray, y: np.ndarray,
                target_var: float = TARGET_VAR,
                top_k_corr: int = 3) -> tuple[list[int], list[int]]:
    """Return two PC index lists (0-indexed):

        var_set : smallest set of leading PCs whose CUMULATIVE explained_variance
                  ratio ≥ target_var.  This is the standard fPCA convention
                  ("how many PCs explain X% of the kinematic signal").
                  Note: this is variance of the *strides themselves*, not of
                  peak velocity — but PC1 of stride kinematics almost always
                  dominates and is what is meant by "1 or 2 PCs explain
                  most of the movement".

        cor_set : top `top_k_corr` PCs by |Pearson r| with peak velocity —
                  i.e. the PCs whose stride-shape variation tracks how fast
                  the athlete is.  These are the ones interesting for sprint
                  performance interpretation.
    """
    evr = pca.explained_variance_ratio_
    n_pc = len(evr)
    cum  = np.cumsum(evr)
    n_var = int(np.searchsorted(cum, target_var) + 1)
    var_set = list(range(min(n_var, n_pc)))

    rs = np.array([abs(pearsonr(scores[:, k], y)[0]) for k in range(n_pc)])
    cor_set = np.argsort(-rs)[:top_k_corr].tolist()
    return var_set, cor_set


# Compute per-range PC selections
range_pc_sel: dict[str, dict[str, list[int]]] = {}
for range_name, res in fpca_results.items():
    var_set, cor_set = _select_pcs(res["pca"], res["scores"], res["y"])
    cum_v = res["pca"].explained_variance_ratio_[var_set].sum() * 100
    union = sorted(set(var_set) | set(cor_set))
    print(f"      {range_name}: var-set={[p+1 for p in var_set]} "
          f"(cum.var={cum_v:.1f}%)  |  cor-set={[p+1 for p in cor_set]}  "
          f"→ render {[p+1 for p in union]}")
    range_pc_sel[range_name] = {
        "variance":    var_set,
        "correlation": cor_set,
        "union":       union,
    }


def _render_pc_overlay(range_name, pc_idx):
    """Render a 2×3 PC overlay figure — participants ranked by score on PC pc_idx
    (1-indexed).  Uses cohort-detected phase frames so the columns actually
    correspond to Touchdown / Mid-Stance / Toe-Off."""
    res = fpca_results[range_name]
    pids   = res["pids"]
    scores = res["scores"]
    k = pc_idx - 1
    r_k  = res["corrs"][k]
    ev_k = res["pca"].explained_variance_ratio_[k] * 100

    pc_scores = scores[:, k]
    order = np.argsort(pc_scores)
    if r_k >= 0:
        slow_idx, fast_idx = order[0],  order[-1]
    else:
        slow_idx, fast_idx = order[-1], order[0]
    med_idx = order[len(order) // 2]

    slow_pid, med_pid, fast_pid = pids[slow_idx], pids[med_idx], pids[fast_idx]
    v_slow = velocity_traces[slow_pid]["peak_vel"]
    v_med  = velocity_traces[med_pid]["peak_vel"]
    v_fast = velocity_traces[fast_pid]["peak_vel"]
    sc_slow, sc_med, sc_fast = pc_scores[slow_idx], pc_scores[med_idx], pc_scores[fast_idx]

    per_pid = range_data[range_name]
    s_slow = per_pid[slow_pid] * SCALE_MM
    s_med  = per_pid[med_pid]  * SCALE_MM
    s_fast = per_pid[fast_pid] * SCALE_MM

    phase_pts = _cohort_phase_points(range_name, pids)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5), constrained_layout=True)
    rows  = ["Sagittal", "Frontal"]
    codes = ["xz",       "yz"]

    for row, (plane_name, plane_code) in enumerate(zip(rows, codes)):
        for col, (phase_name, frame_idx) in enumerate(phase_pts):
            ax = axes[row, col]
            for stride, color, alpha, lw in [
                (s_slow, COLOR_SLOW, 0.75, 1.8),
                (s_med,  COLOR_MED,  0.55, 1.2),
                (s_fast, COLOR_FAST, 0.75, 1.8),
            ]:
                pts = stride[frame_idx].copy()
                centre = pts[MK_PELVIS].copy(); centre[2] = 0.0
                pts = pts - centre
                pts[:, 2] = pts[:, 2] - pts[:, 2].min()
                _draw_overlay(ax, pts, color, alpha, lw, plane=plane_code)

            ax.axhline(0, color="#999999", lw=0.8, alpha=0.6, zorder=0)
            if row == 0:
                ax.set_title(phase_name, fontsize=11, pad=6)
            if col == 0:
                ax.set_ylabel(plane_name, fontsize=11, fontweight="bold")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25); ax.tick_params(labelsize=8)
            if plane_code == "xz":
                ax.set_xlim(-900, 1100); ax.set_ylim(-50, 2050)
            else:
                ax.set_xlim(-800, 800);  ax.set_ylim(-50, 2050)

    pretty = range_name.replace("_", " ").replace("steps", "Steps")
    sel_tag = _selection_tag(range_name, pc_idx)
    fig.suptitle(
        f"Acceleration fPCA — {pretty} — PC{pc_idx} Overlay  (n = {len(pids)})\n"
        f"PC{pc_idx}: {ev_k:.1f}% variance  •  r = {r_k:+.2f} with peak velocity"
        + (f"  •  {sel_tag}" if sel_tag else "")
        + f"\nAll skeletons scaled to {TARGET_HEIGHT_M:.2f} m (same anthropometrics)",
        fontsize=11, fontweight="bold")

    handles = [
        plt.Line2D([], [], color=COLOR_SLOW, lw=2.5,
                   label=f"Slower sprint  ({v_slow:.1f} m/s  ·  PC{pc_idx}={sc_slow:+.1f})"),
        plt.Line2D([], [], color=COLOR_MED,  lw=2.0,
                   label=f"Median  ({v_med:.1f} m/s  ·  PC{pc_idx}={sc_med:+.1f})"),
        plt.Line2D([], [], color=COLOR_FAST, lw=2.5,
                   label=f"Faster sprint  ({v_fast:.1f} m/s  ·  PC{pc_idx}={sc_fast:+.1f})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=10, frameon=False)

    out = FIG_DIR / f"accel_fpca_{range_name}_PC{pc_idx}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _render_pc_shape_mode(range_name, pc_idx):
    """Render 4-panel ±2 SD shape-mode figure for PC pc_idx (1-indexed).
    Layout: sagittal/frontal × −2 SD/+2 SD, each panel showing the full 101-
    frame stride as a faint motion envelope with the mid-stance frame bold."""
    res = fpca_results[range_name]
    pca   = res["pca"]
    pids  = res["pids"]
    k = pc_idx - 1
    Xs = np.stack([range_data[range_name][p].reshape(-1) for p in pids])
    mu_3d   = Xs.mean(axis=0).reshape(TN_POINTS, 64, 3)
    loading = pca.components_[k].reshape(TN_POINTS, 64, 3)
    pc_sd   = float(np.sqrt(pca.explained_variance_[k]))
    r_k     = res["corrs"][k]
    ev_k    = pca.explained_variance_ratio_[k] * 100

    fast_is_positive = (r_k > 0)
    pos_color = "#3a76b3" if fast_is_positive else "#c44e52"
    neg_color = "#c44e52" if fast_is_positive else "#3a76b3"
    fast_side = "blue"   if fast_is_positive else "red"
    fast_sign = "+"      if fast_is_positive else "−"

    plus_recon  = mu_3d + 2.0 * pc_sd * loading
    minus_recon = mu_3d - 2.0 * pc_sd * loading

    # Pick a "bold frame" = the cohort median mid-stance for this range
    phase_pts = _cohort_phase_points(range_name, pids)
    bold_frame = phase_pts[1][1]  # mid-stance

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 11), constrained_layout=True)
    panel_cfg = [
        (0, 0, "sagittal", "xz", minus_recon, neg_color, "− 2 SD  (sagittal)"),
        (0, 1, "sagittal", "xz", plus_recon,  pos_color, "+ 2 SD  (sagittal)"),
        (1, 0, "frontal",  "yz", minus_recon, neg_color, "− 2 SD  (frontal)"),
        (1, 1, "frontal",  "yz", plus_recon,  pos_color, "+ 2 SD  (frontal)"),
    ]
    for r, c, plane_name, plane_code, recon, color, panel_title in panel_cfg:
        ax = axes[r, c]
        # Motion envelope (faint, all frames)
        for fi in range(TN_POINTS):
            pts = (recon[fi] * SCALE_MM).copy()
            centre = pts[MK_PELVIS].copy(); centre[2] = 0.0
            pts = pts - centre
            pts[:, 2] = pts[:, 2] - pts[:, 2].min()
            _draw_overlay(ax, pts, color, alpha=0.10, lw=0.8, plane=plane_code)
        # Mid-stance bold focal point
        pts = (recon[bold_frame] * SCALE_MM).copy()
        centre = pts[MK_PELVIS].copy(); centre[2] = 0.0
        pts = pts - centre
        pts[:, 2] = pts[:, 2] - pts[:, 2].min()
        _draw_overlay(ax, pts, color, alpha=0.95, lw=1.8, plane=plane_code)

        ax.axhline(0, color="#999999", lw=0.8, alpha=0.6, zorder=0)
        ax.set_title(panel_title, fontsize=11, color=color, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25); ax.tick_params(labelsize=8)
        if plane_code == "xz":
            ax.set_xlim(-900, 1100); ax.set_ylim(-50, 2050)
        else:
            ax.set_xlim(-800, 800);  ax.set_ylim(-50, 2050)

    pretty  = range_name.replace("_", " ").replace("steps", "Steps")
    sel_tag = _selection_tag(range_name, pc_idx)
    fig.suptitle(
        f"Accel fPCA — {pretty}  |  PC {pc_idx}  •  {ev_k:.1f}% variance explained\n"
        f"r = {r_k:+.3f} with peak velocity  ({fast_sign} faster sprinter side shown in {fast_side})"
        + (f"  •  {sel_tag}" if sel_tag else "")
        + f"\nAll skeletons scaled to {TARGET_HEIGHT_M:.2f} m (same anthropometrics)",
        fontsize=11, fontweight="bold")

    out = FIG_DIR / f"accel_fpca_{range_name}_PC{pc_idx}_shape.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _selection_tag(range_name: str, pc_idx: int) -> str:
    """Return a short tag describing why this PC was selected."""
    sel = range_pc_sel[range_name]
    in_v = (pc_idx - 1) in sel["variance"]
    in_c = (pc_idx - 1) in sel["correlation"]
    if in_v and in_c:  return "selected by 95 % kinematic variance & top-3 |r|"
    if in_v:           return "selected by 95 % kinematic variance"
    if in_c:           return "selected by top-3 |r| with peak velocity"
    return ""


for range_name in STEP_RANGES:
    for pc_idx_0 in range_pc_sel[range_name]["union"]:
        pc_idx = pc_idx_0 + 1   # 1-indexed for filename / titles
        op1 = _render_pc_overlay(range_name, pc_idx)
        op2 = _render_pc_shape_mode(range_name, pc_idx)
        print(f"      wrote {op1.name}  +  {op2.name}  ({_selection_tag(range_name, pc_idx)})")

print("[6/6] Done.")
