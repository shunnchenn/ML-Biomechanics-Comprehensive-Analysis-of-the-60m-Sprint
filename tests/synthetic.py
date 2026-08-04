"""Synthetic sprinter with known contact/flight timings, heading and stature.

Used to pin the alignment and event code without touching participant data. The
trial is generated in a lab frame, then rotated by an arbitrary yaw and given a
lateral drift, so anything that recovers the truth must be doing so in spite of
both.
"""
from __future__ import annotations

import numpy as np

from sprint import config as C


def make_trial(n_steps=12, contact=7, flight=7, v=9.0, fs=60,
               stature=1.80, yaw_deg=37.0, drift=0.6, noise=0.002, seed=0):
    """Return (markers (n_frames, 64, 3), truth dict)."""
    rng = np.random.default_rng(seed)
    period = contact + flight
    n = n_steps * period + 1
    t = np.arange(n) / fs
    sl = v / (fs / period)                      # step length from v and step rate

    tds = [k * period for k in range(n_steps)]
    tos = [td + contact for td in tds]

    m = np.zeros((n, C.N_MARKERS, 3))
    pel = np.stack([v * t, np.zeros(n),
                    0.53 * stature + 0.02 * np.sin(2 * np.pi * np.arange(n) / period)], 1)

    foot = {}
    for si, side in enumerate(C.SIDES):
        x, z = np.zeros(n), np.zeros(n)
        written = np.zeros(n, bool)
        for k in [k for k in range(n_steps) if k % 2 == si]:
            td, to = tds[k], tos[k]
            plant = v * t[td] + 0.10 * stature          # lands just ahead of pelvis
            x[td:to + 1] = plant + np.linspace(0, 0.02, to - td + 1)  # slight roll
            z[td:to + 1] = 0.0
            written[td:to + 1] = True
            nxt = min(td + 2 * period, n - 1)            # same foot cycles every 2 steps
            if nxt > to:
                u = (np.arange(to, nxt + 1) - to) / (2 * period)
                # Smoothstep, not linear: the swing foot is near-stationary at
                # toe-off and touchdown and fast in between, as a real one is.
                x[to:nxt + 1] = plant + 2 * sl * (3 * u ** 2 - 2 * u ** 3)
                z[to:nxt + 1] = 0.25 * stature * np.sin(np.pi * u)
                written[to:nxt + 1] = True
        # Hold the first/last defined pose across any unwritten head or tail, so
        # the foot never teleports to the origin and creates a fake extremum.
        w = np.flatnonzero(written)
        x[:w[0]], z[:w[0]] = x[w[0]], z[w[0]]
        x[w[-1] + 1:], z[w[-1] + 1:] = x[w[-1]], z[w[-1]]
        foot[side] = (x, z)

    def put(role, xyz):
        m[:, C.MARKER_IDX[role], :] = xyz[:, None, :]

    put("pelvis", pel)
    put("head", pel + np.c_[np.zeros(n), np.zeros(n), np.full(n, 0.47 * stature)])
    for side, sgn in zip(C.SIDES, (-1, 1), strict=True):
        x, z = foot[side]
        heel = np.stack([x, np.full(n, sgn * 0.09), z], 1)
        put(f"{side}_heel", heel)
        put(f"{side}_toe", heel + np.c_[np.full(n, 0.20), np.zeros(n), np.zeros(n)])
        put(f"{side}_ankle", heel + np.c_[np.zeros(n), np.zeros(n), np.full(n, 0.07)])
        lat = np.full(n, sgn)
        put(f"{side}_knee", (heel + pel) / 2 + np.c_[np.zeros(n), lat * 0.05, np.zeros(n)])
        put(f"{side}_hip", pel + np.c_[np.zeros(n), lat * 0.10, np.zeros(n)])
        put(f"{side}_shoulder",
            pel + np.c_[np.zeros(n), lat * 0.20, np.full(n, 0.32 * stature)])
        put(f"{side}_wrist",
            pel + np.c_[-x * 0.02, lat * 0.24, np.full(n, 0.10 * stature)])

    m += rng.normal(0, noise, m.shape)

    yaw = np.radians(yaw_deg)
    rot = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    m = m @ rot.T
    lateral = np.array([-np.sin(yaw), np.cos(yaw)])
    m[:, :, :2] += (np.linspace(0, drift, n)[:, None] * lateral)[:, None, :]

    return m, dict(fs=fs, td=tds, to=tos, contact_s=contact / fs,
                   flight_s=flight / fs, v=v, step_length=sl, stature=stature,
                   yaw=np.array([np.cos(yaw), np.sin(yaw)]))
