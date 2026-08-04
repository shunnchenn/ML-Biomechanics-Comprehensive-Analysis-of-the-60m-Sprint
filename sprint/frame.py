"""Putting every athlete on the same axes.

Vertical is taken from the raw Xsens Z axis rather than re-derived: the IMU suit
measures gravity directly, so vertical is observed, not inferred from variance,
and cannot rotate into the horizontal plane when an athlete's lateral spread
exceeds their height.

Heading is estimated locally, over the window being analysed, not globally over
the whole 60 m. A single axis fitted to the full run is tilted by lane drift and
by the curved path out of the blocks; pelvis displacement across one step is
immune to both and is sign-unambiguous, because the pelvis always advances.

Translation and scale are removed as well, so what remains is a yaw-only
Procrustes alignment: the only free rotation is the one gravity leaves open.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .io import points

UP = np.array([0.0, 0.0, 1.0])


def stature(markers, idx, frames=slice(0, 10)):
    """Standing height: highest head marker minus lowest foot marker."""
    head = markers[frames][:, idx["head"], 2].max()
    foot = min(markers[frames][:, idx[r], 2].min() for r in C.FOOT_ROLES)
    return float(head - foot)


def heading(markers, idx, lo=0, hi=None):
    """Unit forward vector: horizontal pelvis displacement across [lo, hi)."""
    pel = points(markers, idx, "pelvis")[lo:hi]
    d = pel[-1, :2] - pel[0, :2]
    n = np.linalg.norm(d)
    if n < 1e-9:
        raise ValueError("no pelvis displacement in window; cannot define heading")
    return d / n


def basis(fwd):
    """Right-handed (forward, lateral, up) rotation matrix from a 2-D heading."""
    f = np.array([fwd[0], fwd[1], 0.0])
    lat = np.cross(UP, f)
    return np.stack([f, lat / np.linalg.norm(lat), UP])


def to_local(markers, fwd, origin=None, scale=1.0):
    """Rotate into (forward, lateral, up), re-origin, and divide by scale."""
    out = markers if origin is None else markers - np.asarray(origin)
    return (out @ basis(fwd).T) / scale


def heading_drift(markers, idx, windows):
    """Max angle (deg) between per-window headings — a trial-quality check."""
    vecs = [heading(markers, idx, lo, hi) for lo, hi in windows]
    ang = [np.degrees(np.arccos(np.clip(a @ b, -1, 1)))
           for a in vecs for b in vecs]
    return float(max(ang)) if ang else 0.0
