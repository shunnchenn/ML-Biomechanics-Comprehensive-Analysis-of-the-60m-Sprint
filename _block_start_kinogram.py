"""
_block_start_kinogram.py
~~~~~~~~~~~~~~~~~~~~~~~~

Standalone block-clearance + first-2-steps kinogram for Males / Females.

Fixes two bugs in the prior notebook version (cell 16 of 02_Kinematics_PCA.ipynb):

  1. Front Foot Clearance == Rear Ankle Cross
     Old code searched from `Rear Ankle Cross` onwards for the front HEEL to
     rise ≥ 5 mm above its initial position.  By that time the front heel is
     already typically > 5 mm up, so the loop returned the same frame.
     New code uses the front TOE marker, requires a 25 mm rise, and enforces
     ≥ 2 frame separation from Rear Ankle Cross — toe-off is the true clearance
     event.

  2. Per-participant skeletons rendered at native size — small athletes look
     small in the kinogram, large athletes look large, even though the figure
     is meant to compare technique.
     New code rescales every athlete to TARGET_HEIGHT_M = 1.75 m (apparent
     standing height = max head-Z − min foot-Z) before drawing.  All segment
     proportions are preserved because a single scalar multiplies every joint.

Output:
  Outputs/Figures/kinogram_block_scr_M.png
  Outputs/Figures/kinogram_block_scr_F.png

Usage:
  python _block_start_kinogram.py
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

NB_PATH = Path("notebooks/02_Kinematics_PCA.ipynb")

# ── 1. Load constants + canonical helpers from the notebook ───────────────
print("[1/5] Loading notebook helpers …")
with open(NB_PATH) as f:
    _nb = json.load(f)
ns: dict = {"__name__": "__nb_helpers__"}
for cell_idx in (1, 2, 3, 4):
    src = "".join(_nb["cells"][cell_idx]["source"])
    src = "\n".join(line for line in src.split("\n") if "get_ipython" not in line)
    exec(compile(src, f"<cell {cell_idx}>", "exec"), ns)
globals().update(ns)

# Also need cell 12 (kinogram drawing primitives — SKELETON_BONES,
# draw_skeleton_proj, draw_ground_line, COLOR_SLOW/MED/FAST, _AP/_ML/_VERT).
for cell_idx in (12,):
    src = "".join(_nb["cells"][cell_idx]["source"])
    src = "\n".join(line for line in src.split("\n") if "get_ipython" not in line)
    exec(compile(src, f"<cell {cell_idx}>", "exec"), ns)
globals().update(ns)
print(f"      → loaded {sum(1 for k in ns if not k.startswith('_'))} symbols")

_AP, _ML, _VERT = 0, 1, 2

# ── 2. Velocity traces ────────────────────────────────────────────────────
print("[2/5] Loading velocity traces …")
with open(DATA_DIR / "velocity_traces.pkl", "rb") as f:
    velocity_traces = pickle.load(f)
print(f"      → {len(velocity_traces)} participants")

# ── 3. Helpers (with FIX + scaling) ───────────────────────────────────────
TARGET_HEIGHT_M = 1.75


def _apparent_height_m(raw64: np.ndarray) -> float:
    """Standing-height proxy in metres: max(head_z) − min(foot_z) over trial."""
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
    h = _apparent_height_m(raw64)
    if not np.isfinite(h) or h <= 0:
        return 1.0
    return target_m / h


def detect_block_clearance_events(raw64, fs):
    """Block clearance event detection with FIXED Front Foot Clearance."""
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
    # Rear Foot Clearance: rear heel rises > 5 mm above floor
    for i in range(1, len(rear_heel_z)):
        if rear_heel_z[i] > rear_heel_z[0] + 0.005:
            ev["Rear Foot Clearance"] = i; break
    ev.setdefault("Rear Foot Clearance", 2)
    # Rear Ankle Cross: rear heel AP crosses front knee AP
    front_knee_x = np.mean(raw64[:first_sec, front_knee_mks, _AP], axis=1)
    rear_heel_ap = raw64[:first_sec, rear_mk, _AP]
    for i in range(ev["Rear Foot Clearance"], len(rear_heel_ap)):
        if rear_heel_ap[i] >= front_knee_x[i]:
            ev["Rear Ankle Cross"] = i; break
    ev.setdefault("Rear Ankle Cross", ev["Rear Foot Clearance"] + 5)
    # FIXED Front Foot Clearance: TOE marker, 25 mm threshold, +2 frame minimum
    front_toe_z = raw64[:first_sec, front_toe_mk, _VERT]
    toe_init = front_toe_z[0]
    min_search = ev["Rear Ankle Cross"] + 2
    for i in range(min_search, len(front_toe_z)):
        if front_toe_z[i] > toe_init + 0.025:
            ev["Front Foot Clearance"] = i; break
    ev.setdefault("Front Foot Clearance", ev["Rear Ankle Cross"] + 8)
    return ev, rear_mk, front_mk


def detect_accel_strides(raw64, fs, max_strides=8, start_frame=0):
    """Foot-alternating step contacts, used to find Step 1 and Step 2."""
    r_heel_z = raw64[:, MK_R_HEEL, _VERT]
    l_heel_z = raw64[:, MK_L_HEEL, _VERT]
    r_contacts, _ = find_peaks(-r_heel_z, distance=15, prominence=0.01)
    l_contacts, _ = find_peaks(-l_heel_z, distance=15, prominence=0.01)
    r_contacts = r_contacts[r_contacts >= start_frame]
    l_contacts = l_contacts[l_contacts >= start_frame]
    all_c = sorted(list(r_contacts) + list(l_contacts))
    cwf = [(c, "R" if r_heel_z[c] <= l_heel_z[c] else "L") for c in all_c]
    cleaned = [cwf[0]] if cwf else []
    for c, f in cwf[1:]:
        if c - cleaned[-1][0] > 8:
            cleaned.append((c, f))
    if len(cleaned) >= 2:
        alt = [cleaned[0]]
        for c, f in cleaned[1:]:
            if f != alt[-1][1]:
                alt.append((c, f))
            else:
                prev = alt[-1][0]
                if r_heel_z[c] + l_heel_z[c] < r_heel_z[prev] + l_heel_z[prev]:
                    alt[-1] = (c, f)
        cleaned = alt
    return cleaned[: max_strides + 1]


# ── 4. Extract block + first-2-steps data for each participant ────────────
print("[3/5] Processing C3D trials …")
accel_raw: dict[str, dict] = {}

for fpath in sorted(C3D_DIR.glob("*.c3d")):
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

        # Anthropometric normalisation BEFORE event detection
        hscale = _height_scale(raw64, TARGET_HEIGHT_M)
        raw64  = raw64 * hscale

        blk_ev, rear_mk, front_mk = detect_block_clearance_events(raw64, fs)
        sstart = blk_ev.get("Front Foot Clearance", 0) + int(fs * 0.05)
        steps  = detect_accel_strides(raw64, fs, max_strides=4, start_frame=sstart)
        step_events = [{"step": i+1, "foot": s[1], "Touchdown": s[0]}
                       for i, s in enumerate(steps)]

        accel_raw[pid] = {
            "raw64":         raw64,
            "fs":            fs,
            "block_events":  blk_ev,
            "step_events":   step_events,
            "hscale":        hscale,
        }
    except Exception as e:
        print(f"    SKIP {pid}: {e}")

print(f"      → {len(accel_raw)} participants extracted")

# Diagnostic: confirm Front Foot Clearance != Rear Ankle Cross
n_same = 0
for pid, d in accel_raw.items():
    if d["block_events"].get("Front Foot Clearance") == d["block_events"].get("Rear Ankle Cross"):
        n_same += 1
print(f"      Front Foot Clearance == Rear Ankle Cross collisions: {n_same}/{len(accel_raw)}")


# ── 5. Pick slowest / median / fastest per sex ─────────────────────────────
print("[4/5] Selecting representatives per sex …")
sex_reps: dict[str, dict] = {}
for sex in ("M", "F"):
    pids_sex = [p for p in accel_raw if PARTICIPANT_SEX.get(p) == sex]
    if len(pids_sex) < 3:
        continue
    vels = sorted(((p, velocity_traces[p]["peak_vel"]) for p in pids_sex),
                  key=lambda x: x[1])
    sex_reps[sex] = {
        "slowest": vels[0],
        "median":  vels[len(vels) // 2],
        "fastest": vels[-1],
    }
    print(f"      {sex}: slow={vels[0][1]:.2f} ({vels[0][0]})  "
          f"med={vels[len(vels)//2][1]:.2f} ({vels[len(vels)//2][0]})  "
          f"fast={vels[-1][1]:.2f} ({vels[-1][0]})")


# ── 6. Render kinograms ────────────────────────────────────────────────────
print("[5/5] Rendering kinograms …")

BLOCK_CUE = {
    "Rear Foot Clearance": "First frame rear foot leaves block.\nRear lower-leg goal: 145°\n(less extension = better).",
    "Rear Ankle Cross":    "Rear foot crosses front knee.\nRear lower-leg goal: 87°\n(more extension = better).",
    "Front Foot Clearance":"Front TOE leaves the blocks (≥25 mm rise).\nFront lower-leg goal: 169°.\nTrunk goal: 30° (stay low).",
}
STEP_CUE = {
    "Step 1": "CoG behind front foot at contact.\nBack straight for effective push-off.",
    "Step 2": "Increased hip height vs Step 1.\nComplete knee extension at push-off.",
}

blk_events_ordered = ["Rear Foot Clearance", "Rear Ankle Cross", "Front Foot Clearance"]


def _common_xy_limits(frames_list, plane):
    """Compute common axis limits across all skeletons in a column so the
    sagittal / frontal panels share the same scale."""
    xs, ys = [], []
    for fr in frames_list:
        pts = fr * SCALE_MM
        pelvis = np.mean(pts[MK_PELVIS_ALL], axis=0)
        pts = pts - pelvis
        h = pts[:, _AP] if plane == "sagittal" else pts[:, _ML]
        v = pts[:, _VERT]
        xs.extend(h.tolist()); ys.extend(v.tolist())
    xpad = 100.0; ypad = 100.0
    return (min(xs) - xpad, max(xs) + xpad,
            min(ys) - ypad, max(ys) + ypad)


for sex, reps in sex_reps.items():
    sex_name = "Males" if sex == "M" else "Females"
    pid_slow, v_slow = reps["slowest"]
    pid_med,  v_med  = reps["median"]
    pid_fast, v_fast = reps["fastest"]

    n_cols   = len(blk_events_ordered) + 2     # +Step1 +Step2
    planes   = ["sagittal", "frontal"]
    col_lbls = blk_events_ordered + ["Step 1 TD", "Step 2 TD"]

    fig, axes = plt.subplots(len(planes), n_cols,
                             figsize=(3.6 * n_cols, 5.5 * len(planes)),
                             constrained_layout=True)

    handles_global, labels_global = [], []

    # Pre-compute per-column UNIFIED limits so scales match across the 3 reps
    col_limits = {}
    for col, col_lbl in enumerate(col_lbls):
        col_limits[col] = {}
        for plane in planes:
            frames = []
            for pid in (pid_slow, pid_med, pid_fast):
                d = accel_raw[pid]
                if col < 3:
                    fi = d["block_events"].get(blk_events_ordered[col], 0)
                else:
                    step_idx = col - 3
                    fi = (d["step_events"][step_idx]["Touchdown"]
                          if step_idx < len(d["step_events"]) else 0)
                fi = min(int(fi), d["raw64"].shape[0] - 1)
                frames.append(d["raw64"][fi])
            col_limits[col][plane] = _common_xy_limits(frames, plane)

    # Find GLOBAL y-limits so all panels share the same vertical scale
    y_lo_all = min(c[p][2] for c in col_limits.values() for p in planes)
    y_hi_all = max(c[p][3] for c in col_limits.values() for p in planes)
    # Symmetric x-limits per plane so the centered skeleton sits in the middle
    x_lo_sag = min(col_limits[c]["sagittal"][0] for c in col_limits)
    x_hi_sag = max(col_limits[c]["sagittal"][1] for c in col_limits)
    x_lo_fro = min(col_limits[c]["frontal"][0]  for c in col_limits)
    x_hi_fro = max(col_limits[c]["frontal"][1]  for c in col_limits)
    x_span_sag = max(x_hi_sag - x_lo_sag, 1.0)
    x_span_fro = max(x_hi_fro - x_lo_fro, 1.0)

    for row, plane in enumerate(planes):
        for col, col_lbl in enumerate(col_lbls):
            ax = axes[row, col]
            for pid, vel, color, alpha, lw, role in [
                (pid_slow, v_slow, COLOR_SLOW, 0.65, 1.8, "Slowest"),
                (pid_med,  v_med,  COLOR_MED,  0.45, 1.2, "Median"),
                (pid_fast, v_fast, COLOR_FAST, 0.65, 1.8, "Fastest"),
            ]:
                d = accel_raw[pid]
                lbl = f"{role} ({vel:.1f} m/s)" if (col == 0 and row == 0) else None
                if col < 3:
                    fi = d["block_events"].get(blk_events_ordered[col], 0)
                else:
                    step_idx = col - 3
                    fi = (d["step_events"][step_idx]["Touchdown"]
                          if step_idx < len(d["step_events"]) else 0)
                fi = min(int(fi), d["raw64"].shape[0] - 1)
                draw_skeleton_proj(ax, d["raw64"][fi], plane=plane,
                                   color=color, alpha=alpha, lw=lw, ms=2.0,
                                   label=lbl, scale=SCALE_MM)
            # Ground line — use median rep as reference
            d_ref = accel_raw[pid_med]
            if col < 3:
                ref_fi = d_ref["block_events"].get(blk_events_ordered[col], 0)
            else:
                step_idx = col - 3
                ref_fi = (d_ref["step_events"][step_idx]["Touchdown"]
                          if step_idx < len(d_ref["step_events"]) else 0)
            ref_fi = min(int(ref_fi), d_ref["raw64"].shape[0] - 1)
            draw_ground_line(ax, d_ref["raw64"][ref_fi], plane=plane, scale=SCALE_MM)

            ax.set_aspect("equal"); ax.grid(alpha=0.2, lw=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            # Common scales: every panel shares the same x-span for its plane
            # and the same y-range so heights are comparable across columns.
            if plane == "sagittal":
                cx = 0.5 * (col_limits[col]["sagittal"][0]
                            + col_limits[col]["sagittal"][1])
                ax.set_xlim(cx - x_span_sag / 2, cx + x_span_sag / 2)
            else:
                cx = 0.5 * (col_limits[col]["frontal"][0]
                            + col_limits[col]["frontal"][1])
                ax.set_xlim(cx - x_span_fro / 2, cx + x_span_fro / 2)
            ax.set_ylim(y_lo_all, y_hi_all)

            if row == 0:
                ax.set_title(col_lbl, fontsize=10, fontweight="bold", pad=6)
            if col == 0:
                ax.set_ylabel("Sagittal" if plane == "sagittal" else "Frontal",
                              fontsize=11, fontweight="bold")
                if row == 0:
                    h, lbls = ax.get_legend_handles_labels()
                    handles_global, labels_global = h, lbls
            if row == len(planes) - 1:
                cue = (BLOCK_CUE.get(blk_events_ordered[col], "")
                       if col < 3 else
                       STEP_CUE.get(f"Step {col - 2}", ""))
                if cue:
                    ax.text(0.5, -0.10, cue, transform=ax.transAxes,
                            fontsize=6.5, ha="center", va="top", style="italic",
                            color="#555555",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                                      edgecolor="#cccccc", lw=0.5))

    fig.suptitle(
        f"Block Clearance SCR + First 2 Steps — {sex_name}\n"
        f"Slowest / Median / Fastest overlay    |    Sagittal & Frontal views\n"
        f"All skeletons scaled to {TARGET_HEIGHT_M:.2f} m (same anthropometrics)",
        fontsize=12, fontweight="bold")

    if handles_global:
        fig.legend(handles_global, labels_global,
                   loc="lower center", bbox_to_anchor=(0.5, -0.04),
                   ncol=3, fontsize=9, frameon=True, framealpha=0.9)

    out_path = FIG_DIR / f"kinogram_block_scr_{sex}.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"      wrote {out_path.name}")

print("Done.")
