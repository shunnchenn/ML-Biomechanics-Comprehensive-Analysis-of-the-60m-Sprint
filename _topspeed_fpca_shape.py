"""
_topspeed_fpca_shape.py
~~~~~~~~~~~~~~~~~~~~~~~

Standalone script for top-speed fPCA shape-mode figures.

Re-processes all C3D files for top-speed strides (5 strides centred on peak
velocity), rescales every athlete to TARGET_HEIGHT_M = 1.75 m (same
anthropometrics), runs fPCA, then renders shape-mode kinograms for:

  • Variance set : smallest set of leading PCs whose cumulative kinematic
                   variance ≥ 95 %. With this dataset PC1 alone is usually
                   sufficient — that is the "1 or 2 PCs explain most of the
                   movement" case.

  • Correlation set : top 3 PCs by |Pearson r| with peak velocity — the
                      shape modes that actually distinguish fast vs slow.

The union of the two sets is rendered (de-duplicated), each PC annotated
with the reason it was selected.

Output:
  Outputs/Figures/topspeed_fpca_PC{n}.png       — overlay (red/grey/blue ranked by PC score)
  Outputs/Figures/topspeed_fpca_PC{n}_shape.png — ±2 SD shape-mode kinogram
  Outputs/Figures/topspeed_fpca_scree.png
  Outputs/Data/topspeed_fpca_scores.csv
"""

from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

NB_PATH = Path("notebooks/02_Kinematics_PCA.ipynb")
TARGET_HEIGHT_M = 1.75
TARGET_VAR      = 0.95
TOP_K_CORR      = 3


# ── 1. Load notebook helpers ──────────────────────────────────────────────
print("[1/6] Loading notebook helpers …")
with open(NB_PATH) as f:
    _nb = json.load(f)
ns: dict = {"__name__": "__nb_helpers__"}
for cell_idx in (1, 2, 3, 4):
    src = "".join(_nb["cells"][cell_idx]["source"])
    src = "\n".join(line for line in src.split("\n") if "get_ipython" not in line)
    exec(compile(src, f"<cell {cell_idx}>", "exec"), ns)
globals().update(ns)
print(f"      → loaded {sum(1 for k in ns if not k.startswith('_'))} symbols")

_AP, _ML, _VERT = 0, 1, 2

# ── 2. Velocity traces ────────────────────────────────────────────────────
print("[2/6] Loading velocity traces …")
with open(DATA_DIR / "velocity_traces.pkl", "rb") as f:
    velocity_traces = pickle.load(f)
print(f"      → {len(velocity_traces)} participants")


# ── 3. Scaling helpers ─────────────────────────────────────────────────────
def _apparent_height_m(raw64: np.ndarray) -> float:
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


def time_normalize_stride_local(seg, target_len=TN_POINTS):
    n_in = seg.shape[0]
    if n_in < 3:
        return None
    src_x = np.linspace(0.0, 1.0, n_in)
    tgt_x = np.linspace(0.0, 1.0, target_len)
    out = np.zeros((target_len, 64, 3))
    for m in range(64):
        for a in range(3):
            out[:, m, a] = np.interp(tgt_x, src_x, seg[:, m, a])
    return out


# ── 4. Re-process C3Ds for top-speed strides (scaled to 1.75 m) ───────────
print("[3/6] Processing C3D trials (top-speed strides, scaled to 1.75 m) …")
pid_strides: dict[str, np.ndarray] = {}   # pid → (TN_POINTS, 64, 3) mean stride

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

        # Anthropometric normalisation BEFORE detecting strides / building vector
        hscale = _height_scale(raw64, TARGET_HEIGHT_M)
        raw64  = raw64 * hscale
        mk     = mk    * hscale

        vel, peak_f, peak_v = compute_velocity(mk, fs, raw64=raw64)
        strides, _ = detect_strides(mk, peak_f)
        if not strides:
            continue
        stride_64 = []
        for s0, s1 in strides:
            tn = time_normalize_stride_local(raw64[s0:s1 + 1])
            if tn is not None:
                stride_64.append(tn)
        if not stride_64:
            continue
        pid_strides[pid] = np.mean(stride_64, axis=0)
    except Exception as e:
        print(f"    SKIP {pid}: {e}")

print(f"      → {len(pid_strides)} participants with valid top-speed strides")


# ── 5. fPCA on the scaled top-speed strides ────────────────────────────────
print("[4/6] Running fPCA + scree plot …")
pids = sorted(pid_strides.keys())
X    = np.stack([pid_strides[p].reshape(-1) for p in pids])          # (n_pid, 19392)
y    = np.array([velocity_traces[p]["peak_vel"] for p in pids])
mu   = X.mean(axis=0)
Xc   = X - mu

n_comp = min(15, len(pids) - 1)
pca    = PCA(n_components=n_comp, random_state=RANDOM_SEED)
scores = pca.fit_transform(Xc)
corrs  = np.array([np.corrcoef(scores[:, k], y)[0, 1] for k in range(n_comp)])

# Save scores
df = pd.DataFrame(scores, index=pids, columns=[f"PC{k+1}" for k in range(n_comp)])
df.insert(0, "peak_vel_ms", y)
df.to_csv(DATA_DIR / "topspeed_fpca_scores.csv")

# Scree plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
xs = range(1, n_comp + 1)
ax1.bar(xs, pca.explained_variance_ratio_ * 100, color="#4c72b0", alpha=0.85)
ax1.set(xlabel="PC", ylabel="Variance (%)",
        title="Top-speed fPCA — Per-PC Kinematic Variance")
ax2.bar(xs, np.abs(corrs), color="#e07b54", alpha=0.85)
for k, c in enumerate(corrs):
    ax2.text(k + 1, abs(c) + 0.01, f"{c:+.2f}", ha="center", fontsize=7,
             color="#4c72b0" if c > 0 else "#c44e52")
ax2.set(xlabel="PC", ylabel="|r| with peak velocity",
        title="Top-speed fPCA — Correlation with peak velocity")
ax2.set_ylim(0, max(0.6, np.abs(corrs).max() * 1.25))
plt.tight_layout()
fig.savefig(FIG_DIR / "topspeed_fpca_scree.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"      top-3 |r|={np.abs(corrs)[:3].round(2)}  cumvar(3)={pca.explained_variance_ratio_[:3].sum()*100:.1f}%")


# ── 6. PC selection: 95% variance + top-3 |r| ─────────────────────────────
print("[5/6] Selecting PCs …")
cum = np.cumsum(pca.explained_variance_ratio_)
n_var   = int(np.searchsorted(cum, TARGET_VAR) + 1)
var_set = list(range(min(n_var, n_comp)))
rs      = np.abs(corrs)
cor_set = np.argsort(-rs)[:TOP_K_CORR].tolist()
union   = sorted(set(var_set) | set(cor_set))

cum_v = pca.explained_variance_ratio_[var_set].sum() * 100
print(f"      var-set={[p+1 for p in var_set]} (cum.var={cum_v:.1f}%)")
print(f"      cor-set={[p+1 for p in cor_set]}  →  render {[p+1 for p in union]}")


# ── 7. Rendering helpers ───────────────────────────────────────────────────
sys.path.insert(0, str(SCRIPT_DIR))
from sprint_animation import AXIAL_IDX, RIGHT_IDX, LEFT_IDX

COLOR_SLOW = "#c44e52"
COLOR_MED  = "#666666"
COLOR_FAST = "#3a76b3"


def _draw_overlay(ax, frame_pts, color, alpha, lw, plane="xz"):
    if plane == "xz":
        h, v = frame_pts[:, 0], frame_pts[:, 2]
    else:
        h, v = frame_pts[:, 1], frame_pts[:, 2]
    for poly in AXIAL_IDX + RIGHT_IDX + LEFT_IDX:
        ax.plot(h[poly], v[poly], color=color, lw=lw, alpha=alpha,
                solid_capstyle="round")


def _selection_tag(pc_idx_1: int) -> str:
    in_v = (pc_idx_1 - 1) in var_set
    in_c = (pc_idx_1 - 1) in cor_set
    if in_v and in_c:  return "selected by 95 % kinematic variance & top-3 |r|"
    if in_v:           return "selected by 95 % kinematic variance"
    if in_c:           return "selected by top-3 |r| with peak velocity"
    return ""


def _phase_pts():
    """Stride-cycle convention: 0 = touchdown, 50 = mid-stance, 100 = toe-off."""
    return [("Touchdown", 0),
            ("Mid-Stance", TN_POINTS // 4),
            ("Toe-Off",   TN_POINTS // 2)]


# ── 8. Per-PC overlay figure: rank participants by PC score ───────────────
def _render_pc_overlay(pc_idx_1: int) -> Path:
    k = pc_idx_1 - 1
    r_k  = corrs[k]
    ev_k = pca.explained_variance_ratio_[k] * 100
    pc_sc = scores[:, k]
    order = np.argsort(pc_sc)
    if r_k >= 0:
        slow_idx, fast_idx = order[0], order[-1]
    else:
        slow_idx, fast_idx = order[-1], order[0]
    med_idx = order[len(order) // 2]
    slow_pid, med_pid, fast_pid = pids[slow_idx], pids[med_idx], pids[fast_idx]
    v_slow = velocity_traces[slow_pid]["peak_vel"]
    v_med  = velocity_traces[med_pid]["peak_vel"]
    v_fast = velocity_traces[fast_pid]["peak_vel"]
    sc_slow, sc_med, sc_fast = pc_sc[slow_idx], pc_sc[med_idx], pc_sc[fast_idx]

    s_slow = pid_strides[slow_pid] * SCALE_MM
    s_med  = pid_strides[med_pid]  * SCALE_MM
    s_fast = pid_strides[fast_pid] * SCALE_MM

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5), constrained_layout=True)
    for row, (plane_name, plane_code) in enumerate(zip(["Sagittal", "Frontal"], ["xz", "yz"])):
        for col, (phase_name, frame_idx) in enumerate(_phase_pts()):
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

    sel_tag = _selection_tag(pc_idx_1)
    fig.suptitle(
        f"Top-Speed fPCA — PC{pc_idx_1} Overlay  (n = {len(pids)})\n"
        f"PC{pc_idx_1}: {ev_k:.1f}% variance  •  r = {r_k:+.2f} with peak velocity"
        + (f"  •  {sel_tag}" if sel_tag else "")
        + f"\nAll skeletons scaled to {TARGET_HEIGHT_M:.2f} m (same anthropometrics)",
        fontsize=11, fontweight="bold")

    handles = [
        plt.Line2D([], [], color=COLOR_SLOW, lw=2.5,
                   label=f"Slower sprint  ({v_slow:.1f} m/s  ·  PC{pc_idx_1}={sc_slow:+.1f})"),
        plt.Line2D([], [], color=COLOR_MED,  lw=2.0,
                   label=f"Median  ({v_med:.1f} m/s  ·  PC{pc_idx_1}={sc_med:+.1f})"),
        plt.Line2D([], [], color=COLOR_FAST, lw=2.5,
                   label=f"Faster sprint  ({v_fast:.1f} m/s  ·  PC{pc_idx_1}={sc_fast:+.1f})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=10, frameon=False)

    out = FIG_DIR / f"topspeed_fpca_PC{pc_idx_1}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ── 9. Shape-mode ±2 SD figure ─────────────────────────────────────────────
def _render_pc_shape_mode(pc_idx_1: int) -> Path:
    k = pc_idx_1 - 1
    mu_3d   = mu.reshape(TN_POINTS, 64, 3)
    loading = pca.components_[k].reshape(TN_POINTS, 64, 3)
    pc_sd   = float(np.sqrt(pca.explained_variance_[k]))
    r_k     = corrs[k]
    ev_k    = pca.explained_variance_ratio_[k] * 100

    fast_is_positive = (r_k > 0)
    pos_color = "#3a76b3" if fast_is_positive else "#c44e52"
    neg_color = "#c44e52" if fast_is_positive else "#3a76b3"
    fast_side = "blue"   if fast_is_positive else "red"
    fast_sign = "+"      if fast_is_positive else "−"

    plus_recon  = mu_3d + 2.0 * pc_sd * loading
    minus_recon = mu_3d - 2.0 * pc_sd * loading
    bold_frame  = TN_POINTS // 4

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 11), constrained_layout=True)
    panel_cfg = [
        (0, 0, "xz", minus_recon, neg_color, "− 2 SD  (sagittal)"),
        (0, 1, "xz", plus_recon,  pos_color, "+ 2 SD  (sagittal)"),
        (1, 0, "yz", minus_recon, neg_color, "− 2 SD  (frontal)"),
        (1, 1, "yz", plus_recon,  pos_color, "+ 2 SD  (frontal)"),
    ]
    for r, c, plane_code, recon, color, panel_title in panel_cfg:
        ax = axes[r, c]
        for fi in range(TN_POINTS):
            pts = (recon[fi] * SCALE_MM).copy()
            centre = pts[MK_PELVIS].copy(); centre[2] = 0.0
            pts = pts - centre
            pts[:, 2] = pts[:, 2] - pts[:, 2].min()
            _draw_overlay(ax, pts, color, alpha=0.10, lw=0.8, plane=plane_code)
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

    sel_tag = _selection_tag(pc_idx_1)
    fig.suptitle(
        f"Top-Speed fPCA — PC {pc_idx_1}  •  {ev_k:.1f}% variance explained\n"
        f"r = {r_k:+.3f} with peak velocity  ({fast_sign} faster sprinter side shown in {fast_side})"
        + (f"  •  {sel_tag}" if sel_tag else "")
        + f"\nAll skeletons scaled to {TARGET_HEIGHT_M:.2f} m (same anthropometrics)",
        fontsize=11, fontweight="bold")

    out = FIG_DIR / f"topspeed_fpca_PC{pc_idx_1}_shape.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ── 10. Render all selected PCs ────────────────────────────────────────────
print("[6/6] Rendering per-PC overlay + shape-mode figures …")
for pc_idx_0 in union:
    pc_idx_1 = pc_idx_0 + 1
    op1 = _render_pc_overlay(pc_idx_1)
    op2 = _render_pc_shape_mode(pc_idx_1)
    print(f"      wrote {op1.name}  +  {op2.name}  ({_selection_tag(pc_idx_1)})")

print("Done.")
