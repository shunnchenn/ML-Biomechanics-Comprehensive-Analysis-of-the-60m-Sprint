"""
sprint_animation.py
~~~~~~~~~~~~~~~~~~~

Python rewrite of Chris Vellucci's MATLAB sprint-visualization pipeline.

Reproduces the 4-panel composite animation from
`02 - Code/01_Create_PCA_Matrix.m` + `xsens_scr_video.m`:

  +----------------------------+-----------------------+
  |  Thoracic velocity curve   |  Sagittal skeleton    |
  |  (peak = red, now = blue)  |  (R=red, L=green,     |
  |                            |   axial=black)        |
  +----------------------------+-----------------------+
  |                                                    |
  |   3D track view (60 m, finish line, jet trail)     |
  |                                                    |
  +----------------------------------------------------+

Output: 1278×642 MP4 at 20 fps  (matches Chris's `SB60-001.avi`).

Pipeline (Python equivalent of MATLAB):
  1. ezc3d.load        — reads raw 64×3 marker data + frame rate
  2. detrend + crop    — polyfit X→Y residual, distance-along-line, crop at 60 m
  3. compute velocity  — 7-marker thoracic centroid, 3D speed
  4. render frames     — matplotlib → piped to ffmpeg → mp4

Usage:
  python sprint_animation.py SB60          # one trial
  python sprint_animation.py --all         # every C3D in C3D_DIR
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
from io import BytesIO

import ezc3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────
C3D_DIR = "/Users/shunchen/Desktop/60m Project Folder/31 Trials Data Folder/C3D, XLSX/Sprint Trials in c3d"
OUT_DIR = "/Users/shunchen/Desktop/60m Project Folder/Shun's Sprints Code/Outputs/Animations"

# Xsens 64-marker indices (Python 0-based; MATLAB 1-based - 1)
MK_T12        = 10
R_HEEL        = 52
L_HEEL        = 58
THORAX_MKS_64 = [10, 11, 12, 13, 14, 20, 21]   # T12,T8,Neck,Head,L-shoulder,R-shoulder + chest

SPRINT_DIST_MM = 62500      # 62.5 m crop
FS_DEFAULT     = 60          # Hz

# Bone polylines (verbatim from MATLAB `xsens_scr_video.m`; 1-based)
AXIAL_POLYS = [
    [2,3,5,4,2,6,7,3,7,8,6,8,9,10,11,14,16,21,12,15,11,15,21,15,22,15,12,22,16],
    [22, 11],
    [21, 11],
    [17,18,19,17,20,18,20,19,20,16],
]
RIGHT_POLYS = [
    [21,23,24,29,24,21],
    [21,29,27,29,28,27],
    [24,27,24,28,27],
    [23,27,23,28,27],
    [27,33,28,33,34,33,35,33,34,27],
    [39,40,39,41,39,42,41,40,42,47,40,47,41,47],
    [47,48,47,49],
    [40,48,40,49],
    [41,48,41,49],
    [48,49,53,48,53,54,53,55,53,56,57,56,58,53],
]
LEFT_POLYS = [
    [22,25,26,25,32,22],
    [22,26],
    [30,31,25,30,25,31,25,26,30,26,31,26],
    [32,30,31,30,32,31,32],
    [30,36,30,37,30,38,36,37,31,36,31,37,31,38,31],
    [43,44,43,45,43,46,45,44,46],
    [50,46,50,45,50,44,50],
    [50,51,50,52,50],
    [44,51,44,52,44],
    [45,51,45,52,45],
    [51,52,51,59,52,59,60,59,60,59,62,59,61,62,63,62,64],
]

# Convert all to 0-based, once
def _z(polys): return [[i - 1 for i in poly] for poly in polys]
AXIAL_IDX = _z(AXIAL_POLYS)
RIGHT_IDX = _z(RIGHT_POLYS)
LEFT_IDX  = _z(LEFT_POLYS)

# Colours (matching MATLAB)
COL_AXIAL = "k"
COL_RIGHT = "#d62728"      # red
COL_LEFT  = "#2ca02c"      # green
COL_DOT   = "k"


# ────────────────────────────────────────────────────────────────────────────
# Data pipeline
# ────────────────────────────────────────────────────────────────────────────
def load_c3d(filepath: str):
    """Load raw 64-marker data. Returns (pts_mm, fs)."""
    c   = ezc3d.c3d(str(filepath))
    raw = c['data']['points'][:3].transpose(2, 1, 0)     # (frames, sensors, 3)
    if raw.shape[1] < 64:                                # zero-pad if fewer sensors
        pad = np.zeros((raw.shape[0], 64 - raw.shape[1], 3))
        raw = np.concatenate([raw, pad], axis=1)
    raw = raw[:, :64, :]
    # ensure millimetres
    if np.max(np.abs(raw)) < 50:
        raw = raw * 1000.0
    fs = float(c['header']['points']['frame_rate'])
    return raw, fs


def detrend_and_crop(pts: np.ndarray) -> np.ndarray:
    """Replicate the MATLAB pipeline:
       1. Reset origin to the most-posterior heel at frame 0
       2. Detrend Y as residual of linear fit y(x)
       3. Re-express X as distance along the fit line
       4. Crop at SPRINT_DIST_MM forward distance
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
    over = np.where(pts[:, :, 0] > SPRINT_DIST_MM)
    if over[0].size:
        stop = over[0].min()
        pts  = pts[: stop + 1]

    return pts


def detect_sprint_start(pts: np.ndarray, fs: float,
                         go_threshold_mps: float = 1.0,
                         idle_threshold_mps: float = 0.3) -> int:
    """Return the frame index where sprinting begins.

    1. Compute forward (X) velocity of the 7-marker thoracic centroid.
    2. Find the first frame where it exceeds `go_threshold_mps` (1 m/s).
    3. Walk backward to the last frame below `idle_threshold_mps` (0.3 m/s).
    Falls back to 0 if the pattern is not found.
    """
    thor = pts[:, THORAX_MKS_64, :].mean(axis=1)   # (n, 3)
    vx   = np.gradient(thor[:, 0], 1.0 / fs)        # mm/s forward

    go_thresh   = go_threshold_mps   * 1000.0
    idle_thresh = idle_threshold_mps * 1000.0

    above = np.where(vx > go_thresh)[0]
    if len(above) == 0:
        return 0
    first_above = above[0]
    below_idle  = np.where(vx[:first_above] < idle_thresh)[0]
    return int(below_idle[-1]) if len(below_idle) else 0


def compute_thoracic_velocity(pts: np.ndarray, fs: float) -> np.ndarray:
    """3D speed of the 7-marker thoracic centroid (mm/s).

    Uses np.gradient (Sprint_Ensemble_Viz canonical method) rather than the
    MATLAB diff-of-magnitudes — gives true 3D speed including lateral/vertical
    components.
    """
    dt   = 1.0 / fs
    thor = pts[:, THORAX_MKS_64, :].mean(axis=1)        # (n, 3)
    vx   = np.gradient(thor[:, 0], dt)
    vy   = np.gradient(thor[:, 1], dt)
    vz   = np.gradient(thor[:, 2], dt)
    return np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)         # mm/s


# ────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ────────────────────────────────────────────────────────────────────────────
def _draw_skel_2d(ax, frame_pts, plane="xz", lw=1.2):
    """Draw 2D skeleton on `ax` in given plane.
       plane = "xz" → sagittal (X horiz, Z vert)
       plane = "xy" → frontal  (X horiz, Y vert)
    """
    if plane == "xz":
        h, v = frame_pts[:, 0], frame_pts[:, 2]
    else:
        h, v = frame_pts[:, 0], frame_pts[:, 1]

    ax.scatter(h, v, c=COL_DOT, s=10, zorder=3, linewidths=0)
    for poly in AXIAL_IDX:
        ax.plot(h[poly], v[poly], color=COL_AXIAL, lw=lw, solid_capstyle="round")
    for poly in RIGHT_IDX:
        ax.plot(h[poly], v[poly], color=COL_RIGHT, lw=lw, solid_capstyle="round")
    for poly in LEFT_IDX:
        ax.plot(h[poly], v[poly], color=COL_LEFT,  lw=lw, solid_capstyle="round")


def _draw_skel_3d(ax, frame_pts, lw=1.0):
    """Draw 3D skeleton on a `mpl_toolkits.mplot3d.Axes3D` instance."""
    x, y, z = frame_pts[:, 0], frame_pts[:, 1], frame_pts[:, 2]
    ax.scatter(x, y, z, c=COL_DOT, s=8, zorder=3, linewidths=0)
    for poly in AXIAL_IDX:
        ax.plot(x[poly], y[poly], z[poly], color=COL_AXIAL, lw=lw)
    for poly in RIGHT_IDX:
        ax.plot(x[poly], y[poly], z[poly], color=COL_RIGHT, lw=lw)
    for poly in LEFT_IDX:
        ax.plot(x[poly], y[poly], z[poly], color=COL_LEFT,  lw=lw)


# ────────────────────────────────────────────────────────────────────────────
# Main render
# ────────────────────────────────────────────────────────────────────────────
def render_video(c3d_path: str, out_path: str, fps: int = 20,
                 dpi: int = 100, every: int = 1) -> None:
    """Build a 4-panel animation and write it to `out_path` as MP4.

    `every`=N renders every Nth frame (speeds up for testing).
    """
    pid          = os.path.splitext(os.path.basename(c3d_path))[0]
    pts, fs      = load_c3d(c3d_path)
    pts          = detrend_and_crop(pts)
    # ── Trim to sprint start (drop idle pre-start frames) ─────────────────
    start_frame  = detect_sprint_start(pts, fs)
    if start_frame > 0:
        pts = pts[start_frame:]
    vel          = compute_thoracic_velocity(pts, fs)            # mm/s
    n_frames     = len(pts)
    time         = np.arange(n_frames) / fs

    max_vel_mps  = float(vel.max()) / 1000.0
    peak_frame   = int(np.argmax(vel))
    peak_time    = time[peak_frame]

    # Finish time = T12 X passes 62.5 m
    t12_x        = pts[:, MK_T12, 0]
    fin_idx      = np.where(t12_x > SPRINT_DIST_MM)[0]
    finish_time  = time[fin_idx[0]] if fin_idx.size else time[-1]

    # Velocity → jet colour index for trailing path
    rel          = vel / max(vel.max(), 1e-9)
    rel_idx      = np.clip(np.ceil(rel * 9999).astype(int), 1, 9999)

    # ── Pre-compute path-view axis limits ─────────────────────────────────
    x_lim_path = (float(pts[:, :, 0].min()), 66000.0)
    y_lim_path = (float(pts[:, :, 1].min()), float(pts[:, :, 1].max()))
    z_lim_path = (float(pts[:, :, 2].min()), float(pts[:, :, 2].max()))

    # ── ffmpeg pipe ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "image2pipe", "-vcodec", "png", "-r", str(fps), "-i", "-",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", "-preset", "fast",
        "-vf", "scale=ceil(iw/2)*2:ceil(ih/2)*2",     # ensure even dimensions
        out_path,
    ]
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # ── Figure (size = 1278×642 to match MATLAB AVI) ─────────────────────
    fig = plt.figure(figsize=(12.78, 6.42), dpi=dpi, facecolor="white")
    # Top row: velocity curve + sagittal skeleton
    ax_vel  = fig.add_axes([0.06, 0.56, 0.42, 0.35])
    ax_side = fig.add_axes([0.55, 0.56, 0.42, 0.35])
    # Bottom row: full-width top-down track view (X forward, Y lateral)
    ax_path = fig.add_axes([0.05, 0.10, 0.92, 0.32])
    fig.suptitle(f"{pid}  •  Max V = {max_vel_mps:.2f} m/s  •  Time to 62.5 m = {finish_time:.2f} s",
                 fontsize=12, fontweight="bold", y=0.97)

    frames_to_render = range(0, n_frames - 1, every)
    n_render         = len(frames_to_render)
    print(f"  Rendering {n_render} frames @ {fps} fps → {out_path}")

    for fi, n in enumerate(frames_to_render):
        # ── Panel 1: velocity curve ──────────────────────────────────────
        ax_vel.clear()
        ax_vel.plot(time, vel, color="k", lw=1.4)
        ax_vel.scatter([peak_time],    [vel[peak_frame]],
                       c=COL_RIGHT, s=80, edgecolors="k", zorder=5)
        ax_vel.scatter([time[n]],      [vel[n]],
                       c="#1f77b4",   s=80, edgecolors="k", zorder=5)
        # Stats in lower-right corner (won't collide with rising curve)
        info = (f"Max V: {max_vel_mps:.2f} m/s\n"
                f"Finish: {finish_time:.2f} s\n"
                f"Now: {time[n]:.2f} s  ({vel[n]/1000:.2f} m/s)")
        ax_vel.text(0.98, 0.04, info, transform=ax_vel.transAxes,
                    fontsize=9, ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#888888", lw=0.6, alpha=0.9))
        ax_vel.set_xlabel("Time (sec)", fontsize=9)
        ax_vel.set_ylabel("Thoracic Velocity (mm/s)", fontsize=9)
        ax_vel.set_xlim(0, time.max())
        ax_vel.set_ylim(0, vel.max() * 1.08)
        ax_vel.tick_params(labelsize=8)
        ax_vel.grid(True, alpha=0.3)

        # ── Panel 2: sagittal skeleton (X horiz, Z vert), T12-centred ────
        ax_side.clear()
        _draw_skel_2d(ax_side, pts[n], plane="xz")
        cx = pts[n, MK_T12, 0]
        ax_side.set_xlim(cx - 1500, cx + 1500)
        ax_side.set_ylim(pts[:, :, 2].min() - 50, pts[:, :, 2].max() + 50)
        ax_side.set_aspect("equal", adjustable="box")
        ax_side.set_xlabel("X (mm)", fontsize=9)
        ax_side.set_ylabel("Z (mm)", fontsize=9)
        ax_side.tick_params(labelsize=8)
        ax_side.grid(True, alpha=0.3)

        # ── Panel 3: top-down track view (X forward, Y lateral) ──────────
        ax_path.clear()
        # 0 and 60 m markers
        ax_path.axvline(0,              color="#888888", lw=1, ls="--")
        ax_path.axvline(SPRINT_DIST_MM, color="k", lw=2)
        ax_path.text(SPRINT_DIST_MM, y_lim_path[1] * 0.85, " 62.5 m",
                     fontsize=9, fontweight="bold")
        # Lane background
        lane_y = max(abs(y_lim_path[0]), abs(y_lim_path[1])) * 1.2
        ax_path.axhspan(-lane_y, lane_y, color="#f5f5f5", zorder=0)

        # trailing path coloured by velocity (jet)
        if n > 1:
            traj = pts[: n + 1, MK_T12, :]
            seg_colors = cm.jet(rel_idx[: n] / 9999.0)
            # Plot in segments for color gradient
            for i in range(1, len(traj)):
                ax_path.plot(traj[i-1:i+1, 0], traj[i-1:i+1, 1],
                             color=seg_colors[i-1], lw=2.5, solid_capstyle="round")
            # Red dot at peak frame (if we've passed it)
            if n >= peak_frame:
                ax_path.scatter([pts[peak_frame, MK_T12, 0]],
                                [pts[peak_frame, MK_T12, 1]],
                                c=COL_RIGHT, s=100, edgecolors="k", zorder=10,
                                label="Peak V")
            # Blue dot at current position
            ax_path.scatter([pts[n, MK_T12, 0]], [pts[n, MK_T12, 1]],
                            c="#1f77b4", s=80, edgecolors="k", zorder=10,
                            label="Now")

        ax_path.set_xlim(-500, 66000)
        ax_path.set_ylim(-lane_y, lane_y)
        ax_path.set_xlabel("X — forward (mm)", fontsize=9)
        ax_path.set_ylabel("Y — lateral (mm)", fontsize=9)
        ax_path.tick_params(labelsize=8)
        ax_path.grid(True, alpha=0.3)
        ax_path.set_title("Sprint Track  (colour = thoracic velocity)",
                          fontsize=9, pad=2)

        # ── Push frame to ffmpeg ─────────────────────────────────────────
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
        pipe.stdin.write(buf.getvalue())

        if (fi + 1) % 50 == 0 or fi == n_render - 1:
            print(f"    {fi+1}/{n_render}", end="\r", flush=True)

    plt.close(fig)
    pipe.stdin.close()
    pipe.wait()
    print(f"\n  ✓ {os.path.basename(out_path)}")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", nargs="?",
                   help="Participant ID (e.g. SB60) or '--all' for every C3D in C3D_DIR")
    p.add_argument("--all",   action="store_true", help="Process every C3D in C3D_DIR")
    p.add_argument("--every", type=int, default=1,
                   help="Render every Nth frame (default 1 = full speed)")
    p.add_argument("--fps",   type=int, default=20, help="Output frame rate")
    return p.parse_args()


def main():
    args = _parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    all_files = sorted(f for f in os.listdir(C3D_DIR) if f.endswith(".c3d"))

    if args.all or args.target == "--all":
        files = all_files
    elif args.target:
        files = [f for f in all_files if args.target.upper() in f.upper()]
        if not files:
            print(f"No C3D matching '{args.target}'")
            sys.exit(1)
    else:
        print("Specify a target (e.g. SB60) or --all"); sys.exit(1)

    print(f"Processing {len(files)} trial(s) → {OUT_DIR}")
    failures = []
    for i, fname in enumerate(files, 1):
        pid     = os.path.splitext(fname)[0]
        in_path  = os.path.join(C3D_DIR, fname)
        out_path = os.path.join(OUT_DIR, f"{pid}.mp4")
        print(f"[{i}/{len(files)}] {pid}")
        try:
            render_video(in_path, out_path, fps=args.fps, every=args.every)
        except Exception as exc:
            print(f"  ✗ {pid} FAILED: {exc}")
            failures.append((pid, str(exc)))

    print(f"\nDone. {len(files) - len(failures)}/{len(files)} succeeded.")
    for pid, err in failures:
        print(f"  FAILED  {pid}: {err}")


if __name__ == "__main__":
    main()
