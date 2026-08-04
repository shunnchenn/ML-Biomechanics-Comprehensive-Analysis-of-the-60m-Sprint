"""Kinematic curves on the phase base, and the scalars that go with them.

Angles are sagittal segment angles measured from vertical, positive forward, with
joint angles taken as differences between them. That keeps every quantity signed,
directly readable ("trunk 42 deg forward at touchdown"), and independent of any
marker-set-specific joint-centre model.

Contact and flight *durations* are kept as explicit scalars. Normalising each
phase to 0-100% is what makes athletes comparable, but it also divides out
timing, and timing is a determinant of sprint speed rather than a nuisance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import events, frame
from .io import points

SEGMENTS = ("trunk", "thigh", "shank", "foot")
JOINTS = ("hip", "knee", "ankle")
CURVES = SEGMENTS + JOINTS


def _ang(v, ref="down"):
    """Signed sagittal angle of v from the reference vertical, degrees, + = forward.

    Segments are measured proximal-to-distal against straight down (thigh, shank,
    foot) or pelvis-to-shoulder against straight up (trunk), so every angle reads
    zero in neutral standing and positive means the distal end is forward.
    """
    return np.degrees(np.arctan2(v[..., 0], v[..., 2] if ref == "up" else -v[..., 2]))


def angles(P, idx, side):
    """(n_frames, len(CURVES)) sagittal angles for locally-framed markers P."""
    def g(role):
        return points(P, idx, role)

    a = {
        "trunk": _ang((g("R_shoulder") + g("L_shoulder")) / 2 - g("pelvis"), "up"),
        "thigh": _ang(g(f"{side}_knee") - g(f"{side}_hip")),
        "shank": _ang(g(f"{side}_ankle") - g(f"{side}_knee")),
        "foot": _ang(g(f"{side}_toe") - g(f"{side}_heel")),
    }
    a["hip"] = a["thigh"] + a["trunk"]        # thigh relative to trunk; + = flexion
    a["knee"] = a["thigh"] - a["shank"]       # 0 = fully extended; + = flexion
    a["ankle"] = a["foot"] - a["shank"] - 90  # 0 = neutral; + = dorsiflexion
    return np.stack([a[k] for k in CURVES], axis=-1)


def step_tensor(markers, idx, rows, fwd=None, scale=1.0):
    """Phase-normalise and locally align a set of steps: (n_steps, N_PHASE, n_mk, 3).

    `fwd` fixes one heading across all steps — used for the first four steps, where
    the athlete is still rotating up out of the blocks and that rotation is signal.
    Left as None, each step gets its own heading, which is immune to lane drift.
    """
    out = []
    for r in rows.itertuples():
        P = events.phase_normalise(markers, r)
        f = fwd if fwd is not None else frame.heading(markers, idx, r.td, r.next_td + 1)
        origin = markers[r.td, idx[f"{r.foot}_toe"]].mean(axis=0)
        out.append(frame.to_local(P, f, origin, scale))
    return np.stack(out) if out else np.empty((0, C.N_PHASE, markers.shape[1], 3))


def curve_stack(tensor, idx, rows):
    """(n_steps, N_PHASE, n_curves) angle curves, one per step."""
    return np.stack([angles(t, idx, r.foot)
                     for t, r in zip(tensor, rows.itertuples(), strict=True)])


def scalars(tensor, curves, rows, stature):
    """Per-step scalar features, indexed the same way as `rows`."""
    td, mid, to = 0, C.N_CONTACT // 2, C.N_CONTACT - 1
    a = {k: curves[:, :, i] for i, k in enumerate(CURVES)}
    pel = tensor[:, :, C.MARKER_IDX["pelvis"], :].mean(axis=2)
    wr = np.stack([tensor[:, :, C.MARKER_IDX[f"{s}_wrist"], :].mean(axis=2)
                   for s in C.SIDES])
    heel = np.stack([tensor[i, td, C.MARKER_IDX[f"{r.foot}_heel"], :].mean(axis=0)
                     for i, r in enumerate(rows.itertuples())])

    return pd.DataFrame({
        "t_contact": rows.t_contact.values,
        "t_flight": rows.t_flight.values,
        "duty": rows.duty.values,
        "step_freq": rows.step_freq.values,
        "step_length_norm": rows.step_length.values / stature,
        "trunk_lean_td": a["trunk"][:, td],
        "trunk_lean_to": a["trunk"][:, to],
        "shank_angle_td": a["shank"][:, td],
        "knee_td": a["knee"][:, td],
        "knee_midstance": a["knee"][:, mid],
        "hip_ext_to": a["hip"][:, to],
        "ankle_plantarflex_to": a["ankle"][:, to],
        "com_height_td": pel[:, td, 2],
        "com_drop_contact": pel[:, td, 2] - np.nanmin(pel[:, :C.N_CONTACT, 2], axis=1),
        "foot_strike_ahead": heel[:, 0] - pel[:, td, 0],
        "arm_swing_rom": np.nanmax(np.ptp(wr[:, :, :, 0] - pel[None, :, :, 0], axis=2), axis=0),
    })


def select(df, phase, start=0, peak_frame=None):
    """The rows of a step table belonging to one analysis phase."""
    v = df[df.valid & (df.td >= start)]
    if v.empty:
        return v
    if phase == "accel":
        lo, hi = C.ACCEL_STEPS
        return v.iloc[lo - 1:hi]
    k = int(np.argmin(np.abs(v.td.values - peak_frame)))
    half = C.TOPSPEED_STEPS // 2
    return v.iloc[max(0, k - half):max(0, k - half) + C.TOPSPEED_STEPS]


def build(markers, idx, fs, phase):
    """One participant, one phase -> (mean scalars, mean curves, step table, qa)."""
    vel, dist = events.velocity(markers, idx, fs)
    start = int(np.argmax(vel > 1.0))
    peak = int(np.argmax(vel))
    df = events.step_table(markers, idx, fs)
    rows = select(df, phase, start, peak)
    if rows.empty:
        raise ValueError(f"no valid {phase} steps")

    stat = frame.stature(markers, idx)
    fwd = frame.heading(markers, idx, rows.td.iloc[0], rows.next_td.iloc[-1] + 1) \
        if phase == "accel" else None
    tensor = step_tensor(markers, idx, rows, fwd, stat)
    curves = curve_stack(tensor, idx, rows)
    sc = scalars(tensor, curves, rows, stat)
    return (sc.mean(numeric_only=True), np.nanmean(curves, axis=0), rows,
            dict(events.qa(df, float(vel[peak])), peak_vel=float(vel[peak]),
                 stature=stat, distance_m=float(dist[-1])))
