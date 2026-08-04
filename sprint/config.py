"""Paths, marker maps and analysis constants. Single source of truth."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
C3D_DIR = Path(os.environ.get("SPRINT_C3D_DIR", ROOT / "data" / "c3d"))
DATA_DIR = Path(os.environ.get("SPRINT_DATA_DIR", ROOT / "outputs" / "data"))
FIG_DIR = Path(os.environ.get("SPRINT_FIG_DIR", ROOT / "outputs" / "figures"))

FS = 60                       # fallback capture rate (Hz); the c3d header wins
SEED = 42
TRIM_M = 62.0                 # keep the first 62 m of forward travel
EXCLUDED_PIDS = ("SB17",)     # T8 marker lost tracking

# ── Phase base ───────────────────────────────────────────────────────────────
# Every step becomes 0-100% contact followed by 0-100% flight. 20+20 and not 101:
# at 60 Hz a top-speed contact is ~6-7 raw frames, so a denser grid would
# manufacture resolution the data does not hold. Phases shorter than
# MIN_RAW_FRAMES are emitted as NaN rather than interpolated.
N_CONTACT = 20
N_FLIGHT = 20
N_PHASE = N_CONTACT + N_FLIGHT
MIN_RAW_FRAMES = 3

ACCEL_STEPS = (1, 4)          # "first four steps" out of the blocks
TOPSPEED_STEPS = 6            # steps averaged around peak velocity
FAST_SLOW_Q = 1 / 3           # tertile split for the fast-vs-slow contrast

# Physiological QA bounds — a build that violates these is reported, not used silently.
GCT_BOUNDS = (0.07, 0.30)     # s
DUTY_BOUNDS = (0.15, 0.55)    # contact / (contact + flight)
V_SLSF_TOL = 0.10             # |v - SL*SF| / v

# ── Marker roles ─────────────────────────────────────────────────────────────
# Resolved from c3d POINT/LABELS by name (LABELS below); a tuple means "midpoint
# of these". MARKER_IDX is the numeric fallback for arrays that no longer carry
# labels (the committed .npy stride vectors, synthetic test fixtures).
#
# Every index below is confirmed two independent ways against
# outputs/data/accel_participant_vectors.npy: lateral-coordinate sign (right side
# negative) and membership of the RIGHT_POLYS / LEFT_POLYS bone lists in
# sprint_animation.py. Heel/toe/knee/ankle additionally match the constants the
# previous pipeline used.
LABELS = {
    "pelvis": "pHipOrigin",
    "head": ("pRightAuricularis", "pLeftAuricularis"),
    "R_shoulder": "pRightAcromion", "L_shoulder": "pLeftAcromion",
    "R_wrist": ("pRightWristLateral", "pRightWristMedial"),
    "L_wrist": ("pLeftWristLateral", "pLeftWristMedial"),
    "R_hip": "pRightGreaterTrochanter", "L_hip": "pLeftGreaterTrochanter",
    "R_knee": ("pRightKneeLatEpicondyle", "pRightKneeMedEpicondyle"),
    "L_knee": ("pLeftKneeLatEpicondyle", "pLeftKneeMedEpicondyle"),
    "R_ankle": ("pRightLatMalleolus", "pRightMedMalleolus"),
    "L_ankle": ("pLeftLatMalleolus", "pLeftMedMalleolus"),
    "R_heel": "pRightHeelFoot", "L_heel": "pLeftHeelFoot",
    "R_toe": "pRightToe", "L_toe": "pLeftToe",
}

MARKER_IDX = {
    "pelvis": [0, 1, 2, 3, 4, 5, 6],
    "head": [16, 17, 18, 19],
    "R_shoulder": [20], "L_shoulder": [21],
    "R_wrist": [26, 27], "L_wrist": [29, 30],
    "R_hip": [38], "L_hip": [42],
    "R_knee": [39, 40], "L_knee": [43, 44],
    "R_ankle": [47, 48], "L_ankle": [50, 51],
    "R_heel": [52], "L_heel": [58],
    "R_toe": [57], "L_toe": [63],
}
N_MARKERS = 64
FOOT_ROLES = ("R_heel", "L_heel", "R_toe", "L_toe")
SIDES = ("R", "L")
OTHER = {"R": "L", "L": "R"}

PARTICIPANT_SEX = {
    'SB15': 'F', 'SB16': 'F', 'SB17': 'M', 'SB20': 'M', 'SB23': 'F',
    'SB25': 'M', 'SB26': 'M', 'SB50': 'M', 'SB60': 'M', 'SB061': 'M',
    'SB70': 'M', 'SB73': 'M', 'SB74': 'M', 'SB80': 'F', 'SB81': 'F',
    'SB82': 'M', 'SB91': 'F', 'SB92': 'F', 'SB101': 'M', 'SB102': 'F',
    'SB110': 'F', 'SB111': 'M', 'SB112': 'M', 'SB150': 'F', 'SB151': 'F',
    'SB153': 'F', 'SB154': 'M', 'SB155': 'M', 'SB160': 'F', 'SB161': 'F',
    'SB202': 'F',
}
