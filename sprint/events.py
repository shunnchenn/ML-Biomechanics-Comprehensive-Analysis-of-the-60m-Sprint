"""Ground-contact events and the 0-100% contact / 0-100% flight phase base.

Touchdown and toe-off come from the coordinate-based rule of Zeni et al. (2008):
touchdown is where the heel is furthest ahead of the pelvis, toe-off where the
toe is furthest behind it. Each candidate is then snapped to the nearest local
minimum of the marker's own vertical trace, which fixes the few-frame bias the
coordinate rule carries at sprint speeds.

Without force plates these frames are estimates. At 60 Hz one frame is 17 ms,
against a top-speed contact of roughly 100 ms, so per-step contact time carries
about 17% resolution error; averaging over steps reduces it but does not remove
it, and duty-factor results should be read with that in mind.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from . import config as C
from .frame import basis, heading
from .io import points


def velocity(markers, idx, fs):
    """Forward pelvis speed (m/s) and cumulative forward distance (m)."""
    fwd = heading(markers, idx)
    x = points(markers, idx, "pelvis") @ basis(fwd)[0]
    win = min(len(x) // 2 * 2 - 1, max(5, int(0.1 * fs) // 2 * 2 + 1))
    return savgol_filter(x, win, 2, deriv=1, delta=1.0 / fs), x - x[0]


def _runs(foot_z):
    """Inclusive (start, end) frame pairs of each ground-contact run."""
    on = (foot_z <= foot_z.min() + 0.05 * np.ptp(foot_z)).astype(int)
    d = np.diff(on, prepend=0, append=0)
    return np.c_[np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1]


def _snap(cands, runs, edge):
    """Snap each candidate to the nearest start (touchdown) or end (toe-off) of a run.

    The coordinate rule places toe-off several frames late: a foot that has just
    left the ground is near-stationary and keeps falling behind the pelvis after
    it is airborne, so the rule's extremum sits well inside flight. Snapping to
    the contact run itself removes that bias without a tuned search window.
    """
    if not len(runs):
        return np.asarray(cands, dtype=int)
    col = 0 if edge == "first" else 1
    return np.array(sorted({int(runs[np.argmin(np.abs(runs[:, col] - c)), col])
                            for c in cands}), dtype=int)


def _extrema(sig, fs):
    """Peaks of a once-per-stride signal, gated on prominence rather than spacing.

    The minimum spacing is deliberately well below the shortest real stride
    period (~24 frames at 60 Hz); spacing wider than the event being detected is
    what merged consecutive contacts in the previous pipeline. Prominence is
    scaled by a percentile range so one tracking artefact cannot swamp it.
    """
    scale = np.subtract(*np.percentile(sig, [95, 5]))
    pk, _ = find_peaks(sig, distance=max(3, int(0.20 * fs)), prominence=0.25 * scale)
    return pk


def detect(markers, idx, fs):
    """Touchdown and toe-off frames per side, as {'R': (td, to), 'L': (td, to)}."""
    f = basis(heading(markers, idx))[0]
    pel = points(markers, idx, "pelvis") @ f
    out = {}
    for s in C.SIDES:
        heel, toe = (points(markers, idx, f"{s}_{p}") for p in ("heel", "toe"))
        # Lowest point of the foot, so a forefoot strike is handled the same as a
        # rearfoot one — sprinters rarely put the heel down at all.
        runs = _runs(np.minimum(heel[:, 2], toe[:, 2]))
        td = _snap(_extrema(heel @ f - pel, fs), runs, "first")
        to = _snap(_extrema(pel - toe @ f, fs), runs, "last")
        out[s] = (td, to)
    return out


def step_table(markers, idx, fs):
    """One row per step: contact, the flight that follows it, and its timings.

    A step runs touchdown -> toe-off (contact) -> next touchdown of the other
    foot (flight). Steps whose toe-off does not fall inside that span are marked
    invalid rather than silently repaired.
    """
    ev = detect(markers, idx, fs)
    tds = sorted([(int(f), s) for s in C.SIDES for f in ev[s][0]])
    _, dist = velocity(markers, idx, fs)

    rows = []
    for i, (td, side) in enumerate(tds[:-1]):
        nxt = tds[i + 1][0]
        after = ev[side][1][ev[side][1] > td]
        to = int(after[0]) if len(after) else -1
        ok = td < to < nxt
        rows.append(dict(
            step=i + 1, foot=side, td=td, to=to, next_td=nxt,
            t_contact=(to - td) / fs if ok else np.nan,
            t_flight=(nxt - to) / fs if ok else np.nan,
            step_length=float(dist[nxt] - dist[td]),
            valid=ok,
        ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["duty"] = df.t_contact / (df.t_contact + df.t_flight)
    df["step_freq"] = fs / (df.next_td - df.td)
    return df


def _resample(seg, n):
    """Linear resample of (n_frames, n_markers, 3) onto n points."""
    pos = np.linspace(0, len(seg) - 1, n)
    lo = np.floor(pos).astype(int).clip(0, len(seg) - 2)
    w = (pos - lo)[:, None, None]
    return seg[lo] * (1 - w) + seg[lo + 1] * w


def phase_normalise(markers, row):
    """One step on the phase base: (N_PHASE, n_markers, 3).

    Rows 0..N_CONTACT are 0-100% of contact, the rest 0-100% of flight. A phase
    with fewer than MIN_RAW_FRAMES samples is returned as NaN — early-acceleration
    flight can be 2-3 frames at 60 Hz, and interpolating that would invent a curve.
    """
    out = np.full((C.N_PHASE, markers.shape[1], 3), np.nan)
    if not row.valid:
        return out
    for sl, a, b, n in ((slice(0, C.N_CONTACT), row.td, row.to, C.N_CONTACT),
                        (slice(C.N_CONTACT, None), row.to, row.next_td, C.N_FLIGHT)):
        if b - a + 1 >= C.MIN_RAW_FRAMES:
            out[sl] = _resample(markers[a:b + 1], n)
    return out


def qa(df, peak_v):
    """Physiological checks on a step table. Returns {check: bool_passed}."""
    v = df[df.valid]
    sl, sf = v.step_length.mean(), v.step_freq.mean()
    return {
        "n_valid_steps": len(v),
        "gct_in_range": bool(v.t_contact.between(*C.GCT_BOUNDS).all()),
        "duty_in_range": bool(v.duty.between(*C.DUTY_BOUNDS).all()),
        "v_matches_sl_x_sf": bool(abs(peak_v - sl * sf) / peak_v < C.V_SLSF_TOL),
        "mean_gct_s": float(v.t_contact.mean()),
        "mean_duty": float(v.duty.mean()),
        "sl_x_sf_ms": float(sl * sf),
    }
