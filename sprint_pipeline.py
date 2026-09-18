"""
sprint_pipeline.py — the ANALYSIS half of the 60 m sprint pipeline.

Sections 1-6 are preprocessing: load a C3D, clean it, cut it into strides or
steps, turn it into joint angles, and run the fPCA. Sections 9-10 turn that into
a feature table and cross-validated models. Every one of them returns DATA -
arrays, dicts and DataFrames - and nothing here draws anything.

Sections 7 (figures) and 8 (animation) have moved to `sprint_outputs.py`. The
split is deliberate: sections 1-6 are where the testing happens, one step at a
time in `sprint_pipeline.ipynb`, so the output side has no reason to be
interactive. It just renders everything, for everyone, in one call.

    sprint_pipeline.py     sections 1-6, 9, 10   - data only, importable
    sprint_pipeline.ipynb  sections 1-6          - the same code walked through,
                                                   head + shape at every stage
    sprint_outputs.py      sections 7-8          - every figure, every MP4

Height is MEASURED, from 60m Participant Anthropometrics.xlsx. Nothing derives
it from marker positions.

No matplotlib import here on purpose: importing this module can never touch a
notebook's plotting backend.
"""

from __future__ import annotations

from pathlib import Path

import ezc3d
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.interpolate import BSpline



# ==========================================================================
# 1 · Paths, capture settings, marker indices
# ==========================================================================

# ── Paths ───────────────────────────────────────────────────────────────────
# Output directories (figures/, mp4s/) belong to sprint_outputs.py — this module
# writes no files.
HERE = Path(__file__).resolve().parent

DATA_ROOT   = Path("/Users/shunchen/Desktop/60m Project Folder/31 Trials Data Folder")
C3D_DIR     = DATA_ROOT / "C3D, XLSX" / "Sprint Trials in c3d"
ANTHRO_PATH = DATA_ROOT / "60m Participant Anthropometrics.xlsx"

# ── Capture settings ────────────────────────────────────────────────────────
SAMPLING_RATE   = 60          # frames per second
TN_POINTS       = 101         # samples per normalised cycle
TRIM_DISTANCE_M = 62.0        # keep the first 62 m of the run
SPRINT_DIST_M   = 62.5        # finish line drawn in the trial animation
EXCLUDED_PIDS   = ["SB17"]    # T8 marker lost tracking

SPRINT_START_VEL_THRESH = 1.0   # m/s, pelvis forward
SPRINT_START_SUSTAIN    = 5     # frames it must hold for

# Stride/step detection for the ANGLE branch. sprint_helpers uses a 30-frame
# minimum separation, but a top-speed stride at 60 Hz lasts only 27-30 frames,
# so that filter rejects genuine contacts and glues strides together.
STRIDE_MIN_DISTANCE      = 30      # kept only for detect_strides below
STRIDE_PROMINENCE_M      = 0.02    # a contact dips >= 2 cm below its neighbours
ANGLE_STRIDE_MIN_DISTANCE = 18
STRIDE_FRAMES_MIN, STRIDE_FRAMES_MAX = 22, 40   # plausible top-speed stride

# ── Coordinate columns ──────────────────────────────────────────────────────
_AP, _ML, _VERT = 0, 1, 2     # fore-aft, lateral, vertical (Z = gravity, from IMU)

# ── 23-segment indices ──────────────────────────────────────────────────────
IDX_PELVIS, IDX_T8   = 0, 4
IDX_R_FOOT, IDX_L_FOOT = 17, 21
THORAX_IDX = [1, 2, 3, 4, 5, 7, 11]

# ── 64-marker indices ───────────────────────────────────────────────────────
# mk holds segment CENTROIDS, so the real joints live here. "Right Upper Leg"
# in mk is the midpoint of trochanter and knee - neither of the two joints.
MK_PELVIS_ALL = [0, 1, 2, 3, 4, 5, 6]
MK_T12, MK_T8_RAW = 10, 13
MK_HEAD       = [16, 17, 18, 19]
THORAX_MKS_64 = [10, 11, 12, 13, 14, 20, 21]
MK_R_SHOULDER, MK_L_SHOULDER = 20, 21
MK_R_ELBOW, MK_L_ELBOW = [22, 23], [24, 25]
MK_R_WRIST, MK_L_WRIST = [26, 27], [29, 30]
MK_R_HAND,  MK_L_HAND  = [32, 34], [35, 37]
MK_R_HIP,   MK_L_HIP   = 38, 42
MK_R_KNEE,  MK_L_KNEE  = [39, 40], [43, 44]
MK_R_ANKLE, MK_L_ANKLE = [47, 48], [50, 51]
MK_R_HEEL,  MK_L_HEEL  = 52, 58
MK_R_TOE,   MK_L_TOE   = 57, 63
R_HEEL, L_HEEL = MK_R_HEEL, MK_L_HEEL     # aliases used by the trial animation

# ── Colours: ONE convention everywhere ──────────────────────────────────────
COL_FAST, COL_SLOW = "#3a76b3", "#c44e52"   # blue faster, red slower
COL_BODY = "#1f4e79"
COL_AXIAL, COL_RIGHT, COL_LEFT, COL_DOT = "k", "#d62728", "#2ca02c", "k"
LEFT_ALPHA = 0.35                            # left limbs drawn faint


# ==========================================================================
# 2 · Anthropometrics — measured, not derived
# ==========================================================================

# Column in the sheet -> key used in code.
_ANTHRO_COLS = {
    "Body Height": "body_height",   "Shoe Length": "shoe_length",
    "Shoulder Height": "shoulder_height", "Shoulder Width": "shoulder_width",
    "Elbow Span": "elbow_span",     "Wrist Span": "wrist_span",
    "Arm Span": "arm_span",         "Hip Height": "hip_height",
    "Hip Width": "hip_width",       "Knee Height": "knee_height",
    "Ankle Height": "ankle_height",
}


def _load_anthro(path=ANTHRO_PATH):
    """Read the measurement sheet -> {pid: {field: metres}}.

    Height is MEASURED, never derived from marker positions. A marker-derived
    proxy breaks whenever a marker drops out - it read 2.47 m for SB17 against a
    measured 1.76 m - and there is no reason to estimate a quantity that was
    taped.

    Two fixes are applied: three IDs carry trial suffixes (SB70-001 and friends),
    and the sheet is in centimetres. Data-entry errors are corrected in the sheet
    itself rather than patched here, so this function stays a plain reader.

    Raises on any disagreement with the expected cohort, so a bad sheet fails at
    import rather than halfway through an analysis.
    """
    df = pd.read_excel(path)
    table = {}
    for _, row in df.iterrows():
        pid = str(row["Subject Number"]).split("-")[0].strip()   # drop trial suffix
        table[pid] = {key: float(row[col]) / 100.0               # cm -> m
                      for col, key in _ANTHRO_COLS.items()}
    if len(table) != 31:
        raise ValueError(f"expected 31 participants in {path}, found {len(table)}")
    return table


PARTICIPANT_ANTHRO = _load_anthro()


def height_scale_for(pid, target_m=1.75):
    """Uniform scale factor putting `pid` at `target_m`, from measured height.

    One scalar multiplies every joint, so all segment proportions survive. Used
    for DRAWING only - averaging athletes of different sizes would otherwise
    blur the mean skeleton. The fPCA never sees it: angles are size-free.
    """
    anthro = PARTICIPANT_ANTHRO.get(pid)
    if anthro is None:
        raise KeyError(f"no measured anthropometrics for '{pid}'")
    return target_m / anthro["body_height"]


# ==========================================================================
# 3 · Load, align, clean
# ==========================================================================

C3D_TO_XSENS_SEGMENT_MAP = {
    'Pelvis':          'pHipOrigin',
    'L5':              ('pHipOrigin',      'pL5SpinalProcess'),
    'L3':              ('pL5SpinalProcess','pL3SpinalProcess'),
    'T12':             ('pL3SpinalProcess','pT12SpinalProcess'),
    'T8':              ('pT12SpinalProcess','pT8SpinalProcess'),
    'Neck':            ('pT4SpinalProcess','pC7SpinalProcess'),
    'Head':            ('pRightAuricularis','pLeftAuricularis'),
    'Right Shoulder':  ('pIJ', 'pRightAcromion'),
    'Right Upper Arm': ('pRightAcromion', ('pRightArmLateralEpicondyle','pRightArmMedialEpicondyle')),
    'Right Forearm':   (('pRightArmLateralEpicondyle','pRightArmMedialEpicondyle'), ('pRightWristLateral','pRightWristMedial')),
    'Right Hand':      ('pRightWristLateral', 'pRightWristMedial'),
    'Left Shoulder':   ('pIJ', 'pLeftAcromion'),
    'Left Upper Arm':  ('pLeftAcromion', ('pLeftArmLateralEpicondyle','pLeftArmMedialEpicondyle')),
    'Left Forearm':    (('pLeftArmLateralEpicondyle','pLeftArmMedialEpicondyle'), ('pLeftWristLateral','pLeftWristMedial')),
    'Left Hand':       ('pLeftWristLateral', 'pLeftWristMedial'),
    'Right Upper Leg': ('pRightGreaterTrochanter', ('pRightKneeLatEpicondyle','pRightKneeMedEpicondyle')),
    'Right Lower Leg': (('pRightKneeLatEpicondyle','pRightKneeMedEpicondyle'), ('pRightLatMalleolus','pRightMedMalleolus')),
    'Right Foot':      ('pRightLatMalleolus', 'pRightHeelFoot'),
    'Right Toe':       ('pRightHeelFoot', 'pRightToe'),
    'Left Upper Leg':  ('pLeftGreaterTrochanter', ('pLeftKneeLatEpicondyle','pLeftKneeMedEpicondyle')),
    'Left Lower Leg':  (('pLeftKneeLatEpicondyle','pLeftKneeMedEpicondyle'), ('pLeftLatMalleolus','pLeftMedMalleolus')),
    'Left Foot':       ('pLeftLatMalleolus', 'pLeftHeelFoot'),
    'Left Toe':        ('pLeftHeelFoot', 'pLeftToe'),
}


SEGMENTS = [
    'Pelvis', 'L5', 'L3', 'T12', 'T8', 'Neck', 'Head',
    'Right Shoulder', 'Right Upper Arm', 'Right Forearm', 'Right Hand',
    'Left Shoulder', 'Left Upper Arm', 'Left Forearm', 'Left Hand',
    'Right Upper Leg', 'Right Lower Leg', 'Right Foot', 'Right Toe',
    'Left Upper Leg', 'Left Lower Leg', 'Left Foot', 'Left Toe',
]


N_SEGMENTS           = 23       # 23 body segments (pelvis, spine sections, arms, legs, ...)


def _resolve_marker_pos(node, label_to_pos):
    # Look up a marker position by name, or average two positions when given a pair.
    #
    # node         : a string (one marker name) OR a 2-element tuple of nodes
    # label_to_pos : dictionary  {marker_name: 3D_xyz_array}
    # Returns      : (3,) position array, or None if the marker was not found

    # Base case: a single text label — look it up in the dictionary
    if isinstance(node, str):
        return label_to_pos.get(node, None)
    # Recursive case: a pair of nodes — resolve each half and return their midpoint
    if isinstance(node, (list, tuple)) and len(node) == 2:
        a = _resolve_marker_pos(node[0], label_to_pos)
        b = _resolve_marker_pos(node[1], label_to_pos)
        if a is not None and b is not None:
            return (a + b) / 2.0   # midpoint between two 3-D positions
    return None   # could not resolve — return nothing


def c3d_markers_to_segments(raw_markers, labels):
    # Convert 64 raw anatomical markers into 23 body-segment centroids.
    #
    # The C3D file stores 64 reflective dots.  This function groups them into
    # 23 meaningful body parts by averaging the relevant markers for each segment.
    #
    # raw_markers : (n_frames, 64, 3)   [time x marker x xyz]
    # labels      : list of 64 strings, one per marker
    # Returns     : (n_frames, 23, 3)

    # Build a lookup: marker_name -> column_number
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    n_frames = raw_markers.shape[0]
    seg_data = np.zeros((n_frames, N_SEGMENTS, 3))   # output array, starts at zero

    for seg_i, seg_name in enumerate(SEGMENTS):   # loop over every segment
        node = C3D_TO_XSENS_SEGMENT_MAP.get(seg_name)
        if node is None:
            continue   # no mapping defined — leave as zero

        for f in range(n_frames):   # loop over every time frame
            # Build a name->position dictionary just for this single frame
            per_frame = {lbl: raw_markers[f, idx, :] for lbl, idx in label_to_idx.items()}
            pos = _resolve_marker_pos(node, per_frame)
            if pos is not None:
                seg_data[f, seg_i, :] = pos   # store the computed centroid

    return seg_data


def load_c3d(filepath):
    # Load one C3D file and return cleaned marker data.
    #
    # Does three things:
    #   1. Reads the file with ezc3d
    #   2. Fills any missing (NaN) frames by interpolating between neighbours
    #   3. Converts units from mm to metres if needed
    #
    # Returns
    #   marker_data : (n_frames, 23, 3)  metres  — 23 segment centroids
    #   raw_64      : (n_frames, 64, 3)  metres  — all 64 raw markers
    #   fs          : int                Hz      — sampling rate

    c = ezc3d.c3d(str(filepath))
    # C3D stores data as (4, n_markers, n_frames): rows = [x, y, z, confidence]
    points = c['data']['points']
    labels = c['parameters']['POINT']['LABELS']['value']
    fs     = int(c['header']['points']['frame_rate'])

    # Rearrange axes to (n_frames, n_markers, 3) — more intuitive
    raw = points[:3, :, :].transpose(2, 1, 0).copy()

    # Fill gaps: if a marker blinked out (NaN), interpolate from nearby frames
    for mi in range(raw.shape[1]):    # loop over each of the 64 markers
        for ax in range(3):           # loop over x, y, z
            col = raw[:, mi, ax]
            bad = np.isnan(col)       # True where the marker went missing
            if bad.any() and not bad.all():
                good = np.where(~bad)[0]
                col[bad] = np.interp(np.where(bad)[0], good, col[good])

    # Some C3D files store data in millimetres — convert to metres (divide by 1000)
    if np.nanmax(np.abs(raw)) > 100:
        raw /= 1000.0

    marker_data = c3d_markers_to_segments(raw, labels)
    return marker_data, raw, fs


def perform_pca_alignment(marker_data, raw64=None):
    """Chris V.-style alignment (mirrors `pca_align` in (FINAL) Sprint_Ensemble_Viz.ipynb).

    Why this changed: the previous 3D-PCA approach re-derived vertical from
    variance, which silently rotated the gravity axis into a horizontal plane
    whenever lateral spread exceeded height (e.g. SB60, SB91). The Xsens IMU
    already provides a gravity-aligned Z axis on every trial, so vertical is
    fixed — we only need to find the horizontal travel direction.

    Pipeline:
      1. Reset origin to the posterior heel at the first frame.
      2. 2-D SVD on flattened horizontal markers → travel unit vector.
      3. Rotate horizontal plane so travel = display X (+X = direction of motion).
      4. Z passes through unchanged → display Z (up). Matches Kinematics_PCA
         convention: (X=forward, Y=lateral, Z=up).

    Args:
      marker_data : (n_frames, 23, 3) segment centroids
      raw64       : (n_frames, 64, 3) raw markers (required — heel indices live here)

    Returns:
      aligned_mk    : (n_frames, 23, 3) — segments in (forward, lateral, up)
      aligned_raw64 : (n_frames, 64, 3) — raw markers in (forward, lateral, up)
    """
    if raw64 is None:
        raise ValueError("raw64 is required for posterior-heel origin reset.")

    # ── 1. Posterior-heel origin reset (Chris V. lines 46-57) ────────────────
    f0       = raw64[0]
    heel_idx = MK_R_HEEL if f0[MK_R_HEEL, 0] < f0[MK_L_HEEL, 0] else MK_L_HEEL
    origin   = f0[heel_idx].copy()

    mk_c  = marker_data - origin
    r64_c = raw64       - origin

    # ── 2. Travel direction via 2-D SVD on horizontal plane of raw markers ──
    X, Y     = r64_c[:, :, 0], r64_c[:, :, 1]
    xy       = np.column_stack([X.ravel(), Y.ravel()])
    xy_c     = xy - xy.mean(axis=0)
    _, _, Vt = np.linalg.svd(xy_c, full_matrices=False)
    fwd      = Vt[0]
    lat      = np.array([-fwd[1], fwd[0]])     # 90 deg CCW

    def _rotate(arr):
        Xa, Ya, Za = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        return np.stack([Xa * fwd[0] + Ya * fwd[1],
                         Xa * lat[0] + Ya * lat[1],
                         Za], axis=-1)

    mk_aligned  = _rotate(mk_c)
    r64_aligned = _rotate(r64_c)

    # ── 3. Ensure +X = direction of motion ───────────────────────────────────
    centroid_fwd = r64_aligned[:, :, 0].mean(axis=1)
    if centroid_fwd[-1] < centroid_fwd[0]:
        mk_aligned [:, :, 0] *= -1;  mk_aligned [:, :, 1] *= -1
        r64_aligned[:, :, 0] *= -1;  r64_aligned[:, :, 1] *= -1

    return mk_aligned, r64_aligned


def detect_sprint_start(marker_data, fs):
    # Find the frame where sprinting actually begins.
    #
    # The recording starts before the athlete moves, so we skip the standing-still
    # part.  We look for where the pelvis first moves forward faster than 1 m/s
    # and stays that fast for at least 5 consecutive frames.
    #
    # Returns the frame index (integer) of the sprint start.

    vel = np.gradient(marker_data[:, IDX_PELVIS, 0], 1.0 / fs)
    above = vel > SPRINT_START_VEL_THRESH   # True/False array
    for i in range(len(above) - SPRINT_START_SUSTAIN):
        if above[i:i + SPRINT_START_SUSTAIN].all():   # 5 consecutive True values
            return max(0, i - 5)   # step back 5 frames to include the push-off
    return 0   # fallback: use frame 0 if start cannot be detected


def shift_origin(marker_data):
    # Shift the skeleton so the starting foot is at position (0, 0, 0).
    #
    # This makes all participants start at the same spatial point regardless
    # of where they were standing in the lab.

    r_x = marker_data[0, IDX_R_FOOT, 0]
    l_x = marker_data[0, IDX_L_FOOT, 0]
    # Use whichever foot is further back as the origin
    origin = marker_data[0, IDX_R_FOOT if r_x <= l_x else IDX_L_FOOT, :].copy()
    return marker_data - origin   # subtract origin from every marker, every frame


def trim_sprint(marker_data):
    # Trim the trial to the first 62 metres only.

    t8_x = marker_data[:, IDX_T8, 0]   # T8 spine marker x-position over time
    idx  = np.where(t8_x >= TRIM_DISTANCE_M)[0]
    return marker_data[:idx[0]] if len(idx) else marker_data


def compute_velocity(marker_data, fs, raw64=None):
    # Compute 3-D thoracic centroid velocity — mirrors Sprint Viz method.
    #
    # When raw64 is provided (preferred), uses the 7 raw thorax markers
    # (THORAX_MKS_64) and computes the 3-D speed sqrt(vx2+vy2+vz2) so peak
    # values match (FINAL) Sprint_Ensemble_Viz.ipynb.
    # A 10-frame margin avoids gradient edge artefacts.
    #
    # Falls back to 1-D forward-only velocity if raw64 is not supplied.
    #
    # Returns  (velocity_array, peak_frame_index, peak_velocity_value)

    if raw64 is not None:
        dt   = 1.0 / fs
        thor = raw64[:, THORAX_MKS_64, :].mean(axis=1)   # (n, 3) centroid
        vx   = np.gradient(thor[:, 0], dt)
        vy   = np.gradient(thor[:, 1], dt)
        vz   = np.gradient(thor[:, 2], dt)
        vel  = np.sqrt(vx**2 + vy**2 + vz**2)
    else:
        thorax_x = np.mean(marker_data[:, THORAX_IDX, 0], axis=1)
        vel = np.gradient(thorax_x, 1.0 / fs)

    margin = 10
    peak_f = int(np.argmax(vel[margin:len(vel) - margin])) + margin
    return vel, peak_f, float(vel[peak_f])



def find(trial=None):
    """Accept a participant ID, a Path, or None (= first trial). Return a Path."""
    if trial is None:
        return sorted(C3D_DIR.glob("*.c3d"))[0]
    if not isinstance(trial, str):
        return trial
    matches = sorted(C3D_DIR.glob(f"{trial}*.c3d"))
    if not matches:
        raise SystemExit(f"No C3D file found for '{trial}' in {C3D_DIR}")
    return matches[0]


def horizontal_travel_axis(raw64, skip_s=0.0, fs=60):
    """Forward direction taken from where the athlete actually went.

    The old alignment took the forward axis from an SVD of the marker cloud, so
    lateral drift and lead-in frames both tugged on it. Displacement of the
    pelvis from the first frame to the last is a far more stable definition:
    it is simply the direction the athlete travelled.

    Returns a unit 2-vector (AP, ML). Z is not involved at all.
    """
    # 1. Pelvis position at the start and end of the (already trimmed) run.
    #    `skip_s` drops the first stride out of the blocks, where the athlete is
    #    still turning into their lane. Those frames are real sprinting and are
    #    kept in the data - they are only left out of THIS measurement, because
    #    a crooked first stride would tilt the axis everything else is drawn on.
    pelvis = raw64[:, MK_PELVIS_ALL, :].mean(axis=1)
    first  = min(int(round(skip_s * fs)), max(0, len(pelvis) - 2))
    travel = pelvis[-1, [_AP, _ML]] - pelvis[first, [_AP, _ML]]

    # 2. Normalise. If the athlete somehow did not move, fall back to +X.
    norm = float(np.linalg.norm(travel))
    if norm < 1e-6:
        return np.array([1.0, 0.0])
    return travel / norm


def rotate_horizontal(arr, fwd):
    """Rotate an array into (forward, lateral, up) using a horizontal axis.

    IMPORTANT — gravity. The Xsens IMU already supplies a gravity-aligned
    vertical on every trial, so Z is CORRECT as recorded and must never be
    re-derived. This function rotates the horizontal plane only and passes Z
    straight through, exactly like the original `perform_pca_alignment`.
    """
    lat = np.array([-fwd[1], fwd[0]])          # 90 deg CCW, keeps it right-handed
    x, y, z = arr[:, :, _AP], arr[:, :, _ML], arr[:, :, _VERT]
    return np.stack([x * fwd[0] + y * fwd[1],
                     x * lat[0] + y * lat[1],
                     z], axis=-1)              # <- Z untouched


def detect_block_start(raw64, fs, coarse_frame, win_s=1.5):
    """The frame the athlete is still set in the blocks: all four points down.

    In the set position both hands and both feet are on the track. The hands are
    the useful signal, because they leave the ground once and never come back -
    so the LAST frame they are still down is the start of the sprint.

    `detect_sprint_start` gives a coarse anchor from pelvis velocity, and this
    refines it. The search is deliberately limited to a window around that
    anchor: some recordings have 10-20 seconds of lead-in where the athlete
    walks in and settles, and a whole-trial search locks onto an incidental
    hand-lift hundreds of frames too early.
    """
    # 1. Height of the lower hand, measured against its own lowest point in the
    #    window. Using a local reference avoids depending on where the floor is.
    lo = max(0, coarse_frame - int(win_s * fs))
    hi = min(raw64.shape[0], coarse_frame + int(win_s * fs))
    hand = np.minimum(raw64[:, MK_R_HAND, _VERT].mean(axis=1),
                      raw64[:, MK_L_HAND, _VERT].mean(axis=1))
    down = (hand[lo:hi] - hand[lo:hi].min()) < 0.08      # within 8 cm = on the track

    # 2. Last frame still down. If the hands are never down, fall back to the
    #    coarse anchor rather than guessing.
    if not down.any():
        return coarse_frame
    return lo + int(np.where(down)[0].max())


def clean_for_angles(trial=None, settle_s=0.30):
    """Load one trial and clean it for the ANGLE branch. No height scaling.

    Cleaning order is the same as the existing pipeline, with the alignment step
    done twice: a rough pass so the sprint-start detector has a forward axis to
    work with, then a refined pass once the standing lead-in has been removed.
    Refining on clean running only is what squares up the sagittal view.

        load -> rough align -> cut lead-in (+settle) -> refine align
             -> origin reset -> trim 62 m

    `settle_s` drops a little extra time after the detected start.
    `detect_sprint_start` returns `max(0, i - 5)`, a hardcoded step back that
    pulls block-exit frames — where the athlete is still turning into their lane
    — into the alignment window.

    Returns a dict with mk, raw64, fs, and the tilt correction in degrees.
    """
    fpath = find(trial)
    pid   = fpath.stem.split("-")[0].strip()

    # 1. Read the file (gap-fills NaNs, converts mm -> m).
    mk, raw64, fs = load_c3d(fpath)

    # 2. Rough align, so detect_sprint_start has a sensible forward axis.
    mk, raw64 = perform_pca_alignment(mk, raw64)

    # 3. Cut the standing lead-in, back to the moment the athlete is set in the
    #    blocks. The block phase is KEPT: the first steps out of the blocks are
    #    the whole point of the acceleration analysis, and the old velocity-only
    #    start threw away the first 0.22-0.25 s of them.
    coarse = detect_sprint_start(mk, fs)
    start  = min(detect_block_start(raw64, fs, coarse), mk.shape[0] - 2)
    mk, raw64 = mk[start:], raw64[start:]

    # 4. Origin reset: rear foot to (0, 0, 0).
    mk = shift_origin(mk)
    r_x, l_x = raw64[0, MK_R_HEEL, _AP], raw64[0, MK_L_HEEL, _AP]
    raw64 = raw64 - raw64[0, MK_R_HEEL if r_x <= l_x else MK_L_HEEL].copy()

    # 5. Keep the first 62 m.
    mk    = trim_sprint(mk)
    raw64 = raw64[: mk.shape[0]]

    # 6. Refine the forward axis from pelvis travel, measured over the SPAN THAT
    #    SURVIVED trimming. Doing this before the trim leaves a residual tilt,
    #    because cutting the run at 62 m moves the end point the axis was built
    #    from. The old axis is +X by construction after step 2, so the angle
    #    below is exactly how far off square the old view was.
    #    Rotating about the origin leaves (0, 0, 0) where it is, so the origin
    #    reset from step 4 still holds.
    fwd       = horizontal_travel_axis(raw64, skip_s=settle_s, fs=fs)
    tilt_deg  = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
    mk, raw64 = rotate_horizontal(mk, fwd), rotate_horizontal(raw64, fwd)

    # Peak velocity here is UNSCALED, i.e. the athlete's real speed. Nothing on
    # this branch is height-scaled, so it stays comparable across participants.
    _, peak_f, peak_v = compute_velocity(mk, fs, raw64=raw64)

    return {"pid": pid, "fpath": fpath, "fs": fs, "mk": mk, "raw64": raw64,
            "n_frames": int(mk.shape[0]), "tilt_deg": tilt_deg,
            "peak_frame": int(peak_f), "peak_vel_ms": float(peak_v)}


def alignment_report():
    """How far off square was each trial before the pelvis-travel fix?

    One row per trial, worst first. The angle is between the old SVD-derived
    forward axis and the new pelvis-displacement axis, so a large value means
    that trial's exported sagittal view was visibly rotated.
    """
    rows = []
    for fpath in sorted(C3D_DIR.glob("*.c3d")):
        pid = fpath.stem.split("-")[0].strip()
        if pid in EXCLUDED_PIDS:
            continue
        try:
            t = clean_for_angles(fpath)
            rows.append({"pid": pid, "tilt_deg": round(abs(t["tilt_deg"]), 2),
                         "peak_v": round(t["peak_vel_ms"], 2)})
        except Exception as e:
            rows.append({"pid": pid, "tilt_deg": np.nan, "peak_v": np.nan})
    return (pd.DataFrame(rows).sort_values("tilt_deg", ascending=False)
              .reset_index(drop=True))


# ==========================================================================
# 4 · Segmentation — strides at top speed, steps from the blocks
# ==========================================================================

def find_top_speed_strides(mk, peak_frame, n_strides=5):
    """Right-foot contact-to-contact stride windows around peak velocity.

    Why this exists rather than reusing `detect_strides`: that function passes
    `distance=STRIDE_MIN_DISTANCE` (30 frames) to `find_peaks`, but a top-speed
    stride only lasts 27-30 frames. The filter therefore threw away every other
    real contact and returned windows spanning two or three strides. Those
    windows, time-normalised as if they were one cycle, are what flattened the
    averaged curves and made fast/median/slow skeletons look identical.

    Returns (windows, rejected) where windows is a list of (start, end) frame
    pairs whose duration is physiologically plausible.
    """
    # 1. Contacts = minima of right-foot height. Negate so find_peaks sees peaks.
    foot_z = mk[:, IDX_R_FOOT, _VERT]
    contacts, _ = find_peaks(-foot_z,
                             distance=ANGLE_STRIDE_MIN_DISTANCE,
                             prominence=STRIDE_PROMINENCE_M)
    if len(contacts) < 2:
        return [], 0

    # 2. Every consecutive pair is a candidate stride.
    pairs = [(int(a), int(b)) for a, b in zip(contacts[:-1], contacts[1:])]

    # 3. Reject implausible durations. A merged double stride shows up here as
    #    roughly twice the expected period, so this is the safety net.
    good = [(a, b) for a, b in pairs
            if STRIDE_FRAMES_MIN <= (b - a) <= STRIDE_FRAMES_MAX]
    rejected = len(pairs) - len(good)
    if not good:
        return [], rejected

    # 4. Keep the n_strides closest to peak velocity.
    mids  = np.array([(a + b) / 2 for a, b in good])
    order = np.argsort(np.abs(mids - peak_frame))[:n_strides]
    return [good[i] for i in sorted(order)], rejected


def find_first_steps(mk, n_steps=3):
    """The first `n_steps` steps out of the blocks, as (start, end) frame pairs.

    A STEP is contact to contact of ALTERNATE feet, which is the right unit for
    acceleration: coming out of the blocks the two legs are doing different
    jobs, so a same-foot stride would hide the very asymmetry of interest.
    (Top speed is the opposite case, which is why `find_top_speed_strides` uses
    same-foot strides there.)

    Counting is from the sprint start, so step 3 ends at the third ground
    contact after the blocks.

    Returns (windows, rejected). Early steps are longer than top-speed steps -
    the athlete is still slow - so the plausible duration band is wider.
    """
    # 1. Contacts on BOTH feet, since consecutive steps alternate.
    r_z = mk[:, IDX_R_FOOT, _VERT]
    l_z = mk[:, IDX_L_FOOT, _VERT]
    r_c, _ = find_peaks(-r_z, distance=ANGLE_STRIDE_MIN_DISTANCE,
                        prominence=STRIDE_PROMINENCE_M)
    l_c, _ = find_peaks(-l_z, distance=ANGLE_STRIDE_MIN_DISTANCE,
                        prominence=STRIDE_PROMINENCE_M)

    # 2. Merge into one ordered list, tagged with which foot landed.
    contacts = sorted([(int(f), "R") for f in r_c] + [(int(f), "L") for f in l_c])
    if len(contacts) < 2:
        return [], 0

    # 3. Keep only genuine alternations. A repeated foot means one contact was
    #    missed or double-counted, and stepping across it would merge two steps.
    alt = [contacts[0]]
    for f, side in contacts[1:]:
        if side != alt[-1][1]:
            alt.append((f, side))

    # 4. Consecutive pairs are the steps. Early steps run longer than top-speed
    #    ones because the athlete has not wound up yet.
    pairs = [(alt[i][0], alt[i + 1][0]) for i in range(len(alt) - 1)]
    good  = [(a, b) for a, b in pairs if 10 <= (b - a) <= 60]
    return good[:n_steps], len(pairs[:n_steps]) - len(good[:n_steps])


def resample_to_cycle(series, n_points=101):
    """Stretch one stride onto a common 0-100 % axis with `n_points` samples.

    0 % is foot contact and 100 % is the next contact of the SAME foot. Phase,
    not time and not distance, is the only fair axis here: stride frequency
    drifts while an athlete accelerates, so comparing at equal frame number or
    equal metres would compare different parts of the cycle.
    """
    old_x = np.linspace(0.0, 1.0, len(series))
    new_x = np.linspace(0.0, 1.0, n_points)
    return np.interp(new_x, old_x, series)


def stride_cycle_angles(raw64, mk, peak_frame, n_points=101, windows=None):
    """Mean angle curves over a set of stride (or step) windows.

    By default it uses the strides around peak velocity. Pass `windows` to run
    the identical normalisation on any other set of cycles - the acceleration
    steps, for instance - so both phases are treated the same way.

    Returns (mean_curves, n_used, phase_spread) with mean_curves shaped
    (n_points, 13).

    `phase_spread` is the guard against the bug that flattened the old figures.
    It reports how much the moment of peak knee flexion moves between strides.
    If the strides are genuinely the same movement, that moment lands at nearly
    the same percentage every time and averaging is safe. If it scatters, the
    average smears the peaks away and the athlete appears to have no range of
    motion at all.
    """
    if windows is None:
        windows, _ = find_top_speed_strides(mk, peak_frame)
    if not windows:
        return None, 0, np.nan

    angles, _ = compute_joint_angles(raw64)

    # 1. Resample every angle of every stride onto the shared 0-100 % axis.
    per_stride = []
    for a, b in windows:
        cycle = np.column_stack([resample_to_cycle(angles[a:b + 1, j], n_points)
                                 for j in range(angles.shape[1])])
        per_stride.append(cycle)
    per_stride = np.array(per_stride)                    # (n_strides, pts, 8)

    # 2. Phase-consistency check on knee flexion (column 2 = knee_R).
    #    argmax finds peak FLEXION only because knee flexion is positive under
    #    the ISB convention applied in compute_joint_angles. Before that sign
    #    correction this line was locating peak EXTENSION instead.
    peak_phase   = [int(np.argmax(s[:, 2])) for s in per_stride]
    phase_spread = float(max(peak_phase) - min(peak_phase))

    # 3. Average only after that check, so a silent collapse is impossible.
    return per_stride.mean(axis=0), len(windows), phase_spread


def stride_cycle_raw64(raw64, mk, peak_frame, hscale=1.0, n_points=101,
                       windows=None):
    """Mean 64-MARKER cycle, time-normalised the same way as the angles.

    The angle curves are what the fPCA runs on. This returns the matching marker
    positions so a component can be drawn as the real skeleton — the same
    construction the MP4s use — instead of a simplified stick figure.

    `hscale` scales the athlete to a common stature. That is FOR DRAWING ONLY.
    Averaging marker positions across athletes of different sizes blurs the
    result, so the figure needs a common stature to be legible. The fPCA itself
    never sees this: it runs on angles, which are size-free.
    """
    if windows is None:
        windows, _ = find_top_speed_strides(mk, peak_frame)
    if not windows:
        return None

    cycles = []
    for a, b in windows:
        seg = raw64[a:b + 1] * hscale
        # Same resampling as the angles, applied to every marker and axis.
        out = np.zeros((n_points, seg.shape[1], 3))
        for m in range(seg.shape[1]):
            for ax in range(3):
                out[:, m, ax] = resample_to_cycle(seg[:, m, ax], n_points)
        cycles.append(out)
    return np.mean(cycles, axis=0)


# ==========================================================================
# 5 · Joint angles
# ==========================================================================

ANGLE_NAMES = ["hip_R", "hip_L", "knee_R", "knee_L", "ankle_R", "ankle_L",
               "shoulder_R", "shoulder_L", "elbow_R", "elbow_L",
               "trunk_lean", "thigh_separation", "arm_separation"]


def _segment_angle_up(proximal, distal):
    """Same idea for an upward-pointing segment (the trunk).

    0 deg means straight up; positive means the top is ahead — i.e. forward lean.
    """
    d = distal - proximal
    return np.degrees(np.arctan2(d[:, _AP], d[:, _VERT]))


def _sagittal_vec(proximal, distal):
    """Segment as a 2-D vector in the sagittal plane: (fore-aft, vertical)."""
    d = distal - proximal
    return np.column_stack([d[:, _AP], d[:, _VERT]])


def _angle_between(v1, v2):
    """Signed angle turning v1 onto v2, in degrees, positive anticlockwise.

    This is the robust way to define a joint angle. Measuring each segment
    against vertical and subtracting looks equivalent, but it is not: an upper
    arm that swings above the shoulder crosses the +-180 deg wrap point, and the
    subtraction then reports a near-full rotation. That produced a 219 deg
    shoulder range and a 330 deg arm separation, neither of which a person can
    do. The angle BETWEEN two segments never crosses that boundary, because two
    segments joined at a joint can only fold so far.
    """
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot   = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]
    return np.degrees(np.arctan2(cross, dot))


def compute_joint_angles(raw64):
    """Turn a cleaned raw64 array into thirteen joint-angle time series.

    Returns (angles, names) where angles has shape (frames, 13) in degrees and
    names is the matching ordered list, so fPCA loadings can be labelled later.

    Sign conventions, all in the sagittal plane:

        hip_R / hip_L        thigh relative to trunk.
                             POSITIVE = flexion (knee driven forward/up).
        knee_R / knee_L      POSITIVE = flexion (heel toward the buttock).
                             0 = fully straight leg. Follows the ISB reporting
                             standard (Wu et al. 2002 / Grood & Suntay 1983).
                             Measured peak flexion is ~132 deg; published video
                             work reports ~148 deg, so this marker-based
                             definition reads systematically lower.
        ankle_R / ankle_L    foot relative to shank.
                             POSITIVE = dorsiflexion (toes pulled up).
        shoulder_R / _L      upper arm relative to trunk.
                             POSITIVE = arm swung forward.
        elbow_R / elbow_L    POSITIVE = flexion (hand toward the shoulder).
                             0 = fully straight arm.
        trunk_lean           trunk relative to vertical.
                             POSITIVE = leaning forward. 0 = upright.
        thigh_separation     right thigh minus left thigh angle.
                             POSITIVE = right leg ahead of left.
        arm_separation       right upper arm minus left upper arm angle.
                             POSITIVE = right arm ahead of left. Paired with
                             thigh_separation this is the contralateral
                             arm-leg coordination the literature points at.

    Angles are ratios of lengths, so none of these carry units of body size.
    """
    # 1. Joint centres. Paired markers (knee, ankle, elbow, wrist) are averaged
    #    to get the centre of the joint rather than one side of it.
    pelvis = raw64[:, MK_PELVIS_ALL, :].mean(axis=1)
    t8     = raw64[:, MK_T8_RAW, :]
    hip_r,  hip_l  = raw64[:, MK_R_HIP], raw64[:, MK_L_HIP]
    knee_r, knee_l = raw64[:, MK_R_KNEE].mean(axis=1), raw64[:, MK_L_KNEE].mean(axis=1)
    ank_r,  ank_l  = raw64[:, MK_R_ANKLE].mean(axis=1), raw64[:, MK_L_ANKLE].mean(axis=1)
    heel_r, heel_l = raw64[:, MK_R_HEEL], raw64[:, MK_L_HEEL]
    toe_r,  toe_l  = raw64[:, MK_R_TOE],  raw64[:, MK_L_TOE]
    sho_r,  sho_l  = raw64[:, MK_R_SHOULDER], raw64[:, MK_L_SHOULDER]
    elb_r,  elb_l  = raw64[:, MK_R_ELBOW].mean(axis=1), raw64[:, MK_L_ELBOW].mean(axis=1)
    wri_r,  wri_l  = raw64[:, MK_R_WRIST].mean(axis=1), raw64[:, MK_L_WRIST].mean(axis=1)

    # 2. Every segment as a sagittal vector. The trunk is taken DOWNWARD
    #    (T8 -> pelvis) so it is the natural parent for both the thighs and the
    #    upper arms, which both hang off it.
    v_trunk_dn = _sagittal_vec(t8, pelvis)
    v_thigh_r  = _sagittal_vec(hip_r,  knee_r)
    v_thigh_l  = _sagittal_vec(hip_l,  knee_l)
    v_shank_r  = _sagittal_vec(knee_r, ank_r)
    v_shank_l  = _sagittal_vec(knee_l, ank_l)
    v_foot_r   = _sagittal_vec(heel_r, toe_r)
    v_foot_l   = _sagittal_vec(heel_l, toe_l)
    v_uarm_r   = _sagittal_vec(sho_r, elb_r)
    v_uarm_l   = _sagittal_vec(sho_l, elb_l)
    v_farm_r   = _sagittal_vec(elb_r, wri_r)
    v_farm_l   = _sagittal_vec(elb_l, wri_l)

    # 3. Each joint angle is the turn from the parent segment to the child
    #    segment. Working segment-to-segment removes the athlete's global
    #    orientation, so these describe posture rather than which way they were
    #    facing — and unlike differencing two angles-from-vertical, it cannot
    #    wrap.
    hip_R      = _angle_between(v_trunk_dn, v_thigh_r)
    hip_L      = _angle_between(v_trunk_dn, v_thigh_l)
    # Negated to satisfy the ISB convention (Wu et al. 2002, following Grood &
    # Suntay 1983): knee FLEXION is positive with 0 deg = full extension. The
    # knee and the elbow fold in opposite directions, so the same cross-product
    # ordering that makes elbow flexion positive makes knee flexion negative.
    # Without this negation the measured knee ran -131.8 to -16.9 deg, i.e.
    # flexion reported as negative, which contradicts both the docstring above
    # and the reporting standard.
    knee_R     = -_angle_between(v_thigh_r,  v_shank_r)
    knee_L     = -_angle_between(v_thigh_l,  v_shank_l)
    ankle_R    = _angle_between(v_shank_r,  v_foot_r) - 90.0   # neutral near 0
    ankle_L    = _angle_between(v_shank_l,  v_foot_l) - 90.0
    shoulder_R = _angle_between(v_trunk_dn, v_uarm_r)
    shoulder_L = _angle_between(v_trunk_dn, v_uarm_l)
    elbow_R    = _angle_between(v_uarm_r,   v_farm_r)
    elbow_L    = _angle_between(v_uarm_l,   v_farm_l)
    trunk_lean = _segment_angle_up(pelvis, t8)                 # vs vertical

    # 4. Separations are differences of two already-bounded joint angles, so
    #    they stay bounded too.
    angles = np.column_stack([
        hip_R, hip_L, knee_R, knee_L, ankle_R, ankle_L,
        shoulder_R, shoulder_L, elbow_R, elbow_L,
        trunk_lean,
        hip_R - hip_L,              # thigh_separation
        shoulder_R - shoulder_L,    # arm_separation
    ])
    return angles, list(ANGLE_NAMES)


# ==========================================================================
# 6 · Functional PCA on the angle curves
# ==========================================================================

def _bspline_basis(n_points=101, n_basis=15, degree=3):
    """Build the B-spline design matrix for a 0-100 % cycle.

    A B-spline basis is used rather than Fourier because stride frequency drifts
    and joint angles have no clean repeating waveform. Fourier assumes a fixed
    frequency, which is exactly the assumption that does not hold here.
    """
    x = np.linspace(0.0, 100.0, n_points)
    # Knots: evenly spaced inside, repeated `degree+1` times at each end so the
    # basis is well defined right up to the boundary.
    inner = np.linspace(0.0, 100.0, n_basis - degree + 1)
    knots = np.concatenate([np.repeat(inner[0], degree),
                            inner,
                            np.repeat(inner[-1], degree)])
    B = BSpline.design_matrix(x, knots, degree, extrapolate=False).toarray()
    return B


def _second_difference_penalty(n_basis):
    """Penalty matrix that charges for wiggle between neighbouring coefficients."""
    D = np.diff(np.eye(n_basis), n=2, axis=0)
    return D.T @ D


def smooth_angle_curves(curves, penalty=1.0, n_basis=15, degree=3, report=False):
    """Smooth angle curves with a penalised B-spline fit.

    `curves` is (n_points, n_angles) or (n_participants, n_points, n_angles).

    Smoothness is controlled by `penalty` (lambda) rather than by hand-picking a
    basis size: coefficients solve (B'B + lambda*P) c = B'y. A larger penalty
    means a smoother curve.

    When `report` is True the EFFECTIVE DEGREES OF FREEDOM are printed. That is
    the honest measure of how much wiggle survived — near `n_basis` means barely
    smoothed, near 2 means almost a straight line.
    """
    single = (curves.ndim == 2)
    data   = curves[None, ...] if single else curves
    n_pts  = data.shape[1]

    # 1. Basis and penalty do not depend on the data, so build them once.
    B = _bspline_basis(n_pts, n_basis, degree)
    P = _second_difference_penalty(B.shape[1])

    # 2. The smoother matrix maps raw values to fitted values.
    A   = B.T @ B + penalty * P
    hat = B @ np.linalg.solve(A, B.T)

    if report:
        edf = float(np.trace(hat))
        print(f"  effective degrees of freedom = {edf:.1f} "
              f"(basis {B.shape[1]}, penalty {penalty})")

    # 3. Apply it to every participant and every angle.
    out = np.stack([hat @ data[i] for i in range(data.shape[0])])
    return out[0] if single else out


def run_angle_fpca(curves_by_pid, n_components=6, warn_threshold=80.0):
    """Functional PCA across participants, on smoothed joint-angle curves.

    `curves_by_pid` maps pid -> (n_points, n_angles).

    Each angle is standardised to unit variance before the curves are joined
    together. Without that step PC1 becomes "whichever angle happens to swing
    furthest" — knee flexion moves through ~90 deg while trunk lean moves
    through ~10, so the knee would dominate purely by range. That is the same
    kind of artefact as measuring body size, just wearing a different hat.

    Returns a dict with scores, loadings, explained variance and the pids.
    """
    pids   = sorted(curves_by_pid)
    X      = np.stack([curves_by_pid[p] for p in pids])     # (n_pid, pts, ang)
    n_pid, n_pts, n_ang = X.shape

    # 1. Standardise each angle by its own spread across the whole cohort.
    ang_sd = X.std(axis=(0, 1), keepdims=True)
    ang_sd[ang_sd < 1e-9] = 1.0
    Xs = X / ang_sd

    # 2. Flatten each participant to one long row and centre.
    flat = Xs.reshape(n_pid, n_pts * n_ang)
    mean = flat.mean(axis=0)
    Xc   = flat - mean

    # 3. PCA by SVD. With evenly spaced samples this is functional PCA with
    #    uniform quadrature weights.
    n_comp = min(n_components, n_pid - 1)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores   = U[:, :n_comp] * S[:n_comp]
    loadings = Vt[:n_comp].reshape(n_comp, n_pts, n_ang)
    var      = (S ** 2) / (S ** 2).sum() * 100.0

    # 4. Loud sanity check. A component this dominant almost always means a
    #    size or scaling artefact rather than a technique pattern.
    if var[0] > warn_threshold:
        print("!" * 70)
        print(f"WARNING: PC1 explains {var[0]:.1f} % of variance (> {warn_threshold} %).")
        print("The size artefact has probably NOT been removed. Do not interpret")
        print("these components as technique until this is understood.")
        print("!" * 70)

    return {"pids": pids, "scores": scores, "loadings": loadings,
            "explained_var": var[:n_comp], "mean_curve": mean.reshape(n_pts, n_ang),
            "angle_sd": ang_sd.reshape(n_ang), "names": list(ANGLE_NAMES),
            "n_points": n_pts}


def pc_scores_by_angle(fpca, curves_by_pid, pc=1):
    """Split each athlete's PC score into the part contributed by each angle.

    A component score is one number per athlete, and the question "which segment
    is that number coming from?" has an exact answer, because the score is a
    plain sum:

        score(athlete i) = SUM over phase p and angle a of
                           Xc[i, p, a] * loading[p, a]

    where `Xc` is the athlete's cycle after the same two steps the fPCA applied
    -- divide each angle by its cohort SD, subtract the cohort mean curve. Doing
    the inner sum over `p` only, and leaving `a` alone, gives the PARTIAL SCORE
    for one angle. The partial scores add back up to the score exactly; nothing
    is approximated here and no second decomposition is run.

    Returns (partial, total) where partial is (n_pid, n_angles) and total is
    (n_pid,), matching `fpca["scores"][:, pc - 1]`.
    """
    pids    = fpca["pids"]
    X       = np.stack([curves_by_pid[p] for p in pids])       # (n_pid, pts, ang)
    Xc      = X / fpca["angle_sd"] - fpca["mean_curve"]        # the fPCA's own input
    loading = fpca["loadings"][pc - 1]                         # (pts, ang)

    partial = np.einsum("ipa,pa->ia", Xc, loading)
    return partial, partial.sum(axis=1)


def pc_loading_table(fpca, pc=1, curves_by_pid=None, peak_vel=None):
    """The loading vector of a component, one row per angle.

    A component is a CURVE per angle, not a single number, so "the loading of
    the knee" has to be summarised. Three summaries, and they answer different
    questions:

        loading_L2      the size of that angle's loading curve across the whole
                        cycle. This is the bar in `plot_pc_loading_ranking`, and
                        it says how much the component MOVES this angle.
        share_%         loading_L2 squared as a percentage of the total. The
                        loading vector is unit length, so these sum to 100 and
                        read as "this angle is x % of the component".
        peak_loading    the single largest value and the phase it happens at,
                        signed. This is the one that says WHAT the component
                        does -- a positive peak at 20 % of the cycle means a
                        high score goes with more of that angle at 20 %.

    `mean_loading` is signed and often near zero even for a big angle, which is
    not a contradiction: a component that flexes a joint early and extends it
    late has a large curve that averages out.

    Pass `curves_by_pid` and `peak_vel` to add the correlation columns:

        r_partial_vel   correlation between THIS ANGLE'S contribution to the
                        score (see `pc_scores_by_angle`) and peak velocity,
                        across athletes. This is the component's correlation
                        with speed, decomposed by segment.

    Sorted by loading_L2, biggest first.
    """
    k       = pc - 1
    loading = fpca["loadings"][k]                     # (points, angles)
    names   = fpca["names"]

    l2   = np.linalg.norm(loading, axis=0)
    at   = np.argmax(np.abs(loading), axis=0)         # phase of the biggest value
    rows = {
        "angle":          names,
        "loading_L2":     l2.round(3),
        "share_%":        (100 * l2 ** 2 / (l2 ** 2).sum()).round(1),
        "mean_loading":   loading.mean(axis=0).round(4),
        "peak_loading":   loading[at, np.arange(loading.shape[1])].round(3),
        "at_%_of_cycle":  at,
    }

    if curves_by_pid is not None and peak_vel is not None:
        partial, total = pc_scores_by_angle(fpca, curves_by_pid, pc)
        v = np.array([peak_vel[p] for p in fpca["pids"]])
        rows["r_partial_vel"] = np.array(
            [round(float(np.corrcoef(partial[:, j], v)[0, 1]), 3)
             for j in range(partial.shape[1])])
        rows["sd_partial"] = partial.std(axis=0).round(3)

    return (pd.DataFrame(rows)
              .sort_values("loading_L2", ascending=False)
              .reset_index(drop=True))


def correlate_with_velocity(fpca, peak_vel, body_height):
    """Correlate each PC score against peak velocity, and against body height.

    Velocity must be the UNSCALED value. If a trial has been height-scaled to a
    common stature, every length in it is multiplied by that factor and so is
    the velocity derived from it — an athlete scaled from 1.93 m to 1.75 m has
    their speed understated by 9 %. Correlating against that would mostly
    measure the scaling.

    Height is reported alongside on purpose. Angles carry no units of length, so
    the dimensional size artefact is gone, but taller athletes genuinely move
    differently and that is real biomechanics, not a bug. Showing both columns
    lets you see how much size-related signal is left instead of assuming none.
    """
    rows = []
    for k in range(fpca["scores"].shape[1]):
        s = fpca["scores"][:, k]
        rows.append({
            "PC": k + 1,
            "var_%":      round(float(fpca["explained_var"][k]), 1),
            "r_velocity": round(float(np.corrcoef(s, peak_vel)[0, 1]), 3),
            "r_height":   round(float(np.corrcoef(s, body_height)[0, 1]), 3),
        })
    return pd.DataFrame(rows)


def fast_vs_slow_angles(cohort, peak_vel, n=5):
    """Average angle curves for the n fastest vs the n slowest athletes.

    This is the direct question: at the same point in the stride, what do fast
    athletes do differently? It does not depend on fPCA at all, so it is a
    useful check on whether the components are telling the truth.

    Returns (table, fast_curves, slow_curves, fast_ids, slow_ids). The table
    reports, per angle, the biggest gap between the two groups and where in the
    cycle it happens.
    """
    # 1. Rank by real (unscaled) speed and take the two ends.
    ranked = sorted(cohort, key=lambda p: peak_vel[p], reverse=True)
    fast_ids, slow_ids = ranked[:n], ranked[-n:]

    fast = np.mean([cohort[p] for p in fast_ids], axis=0)   # (points, angles)
    slow = np.mean([cohort[p] for p in slow_ids], axis=0)

    # 2. Difference through the cycle, then summarise each angle by its worst gap.
    diff = fast - slow
    rows = []
    for j, name in enumerate(ANGLE_NAMES):
        at = int(np.argmax(np.abs(diff[:, j])))
        rows.append({
            "angle":          name,
            "max_diff_deg":   round(float(diff[at, j]), 1),
            "at_%_of_cycle":  at,
            "mean_diff_deg":  round(float(diff[:, j].mean()), 1),
            "fast_mean_deg":  round(float(fast[:, j].mean()), 1),
            "slow_mean_deg":  round(float(slow[:, j].mean()), 1),
        })
    table = (pd.DataFrame(rows)
               .reindex(pd.DataFrame(rows).max_diff_deg.abs()
                        .sort_values(ascending=False).index)
               .reset_index(drop=True))
    return table, fast, slow, fast_ids, slow_ids


def rank_angles_by_velocity(cohort, peak_vel, top=5):
    """Which joint angles actually predict speed? One row per angle.

    Each angle is summarised two ways across the stride cycle — its average
    value and its range of motion — and each is correlated with real peak
    velocity. Sorted by whichever is stronger.

    This is a plain correlation, not a model. It is here as a sanity check on
    the fPCA loadings: a joint that matters should show up in both.
    """
    pids = sorted(cohort)
    v    = np.array([peak_vel[p] for p in pids])

    rows = []
    for j, name in enumerate(ANGLE_NAMES):
        means = np.array([cohort[p][:, j].mean() for p in pids])
        roms  = np.array([np.ptp(cohort[p][:, j]) for p in pids])
        r_mean = float(np.corrcoef(means, v)[0, 1])
        r_rom  = float(np.corrcoef(roms,  v)[0, 1])
        rows.append({"angle": name,
                     "r_mean_angle": round(r_mean, 3),
                     "r_range_of_motion": round(r_rom, 3),
                     "strongest_abs_r": round(max(abs(r_mean), abs(r_rom)), 3)})

    table = (pd.DataFrame(rows).sort_values("strongest_abs_r", ascending=False)
               .reset_index(drop=True))
    print(f"Top {top} angles by correlation with peak velocity (n = {len(pids)}):")
    for _, r in table.head(top).iterrows():
        print(f"  {r.angle:18} mean r={r.r_mean_angle:+.3f}   ROM r={r.r_range_of_motion:+.3f}")
    return table


# ==========================================================================
# 7 · Figures      →  sprint_outputs.py
# 8 · Animation    →  sprint_outputs.py
# ==========================================================================
#
# Both output sections live in `sprint_outputs.py`, which imports this module.
# The numbering is kept so the two files still read as one pipeline.
#
#     import sprint_outputs as OUT
#     OUT.run_all_outputs()        # every figure, every athlete's MP4
#
# Nothing below draws anything either — sections 9 and 10 return tables, and
# `sprint_outputs.py` draws them.


# ==========================================================================
# 9 · Feature engineering for the ML stage
# ==========================================================================
#
# Everything above answers "what does the movement look like?". This section
# turns that movement into a table a model can read: one row per athlete, one
# column per number. Three jobs, in order.
#
#   1. Clean the TARGET.     The velocity we predict is itself a measurement,
#                            and a noisy one. Smooth it first.
#   2. Clean the FEATURES.    Right and left are not comparable labels across
#                            athletes. Reorder them by range of motion.
#   3. Summarise the curves.  A 101-point curve is not a feature; its range,
#                            its mean and its extremes are.


def smooth_velocity_curve(vel, penalty=30.0, n_basis=60):
    """Penalised B-spline fit of one velocity-time curve.

    Velocity here is a NUMERICAL DERIVATIVE of marker position, and
    differentiating amplifies noise: a 1 mm tremor in a marker at 60 Hz becomes
    6 cm/s of fake velocity. Taking the maximum of that noisy signal then picks
    out the largest upward blip, so the peak is biased HIGH by construction --
    across this cohort by +0.52 m/s on average, and by different amounts for
    different athletes, which means the bias is noise rather than an offset.

    The fix is the same one used on the angle curves: fit a smooth function and
    read the peak off the function instead of off the samples. `_bspline_basis`
    and `_second_difference_penalty` are reused unchanged, so velocity and
    angles are smoothed by identical machinery.

    A denser basis (60) than the angle curves (15) is used because a whole
    60 m run is a much longer signal than one stride cycle. The default penalty
    is the median of a per-athlete GCV selection, and `velocity_penalty_sweep`
    shows the choice barely matters: the peak moves 0.10 m/s across five orders
    of magnitude of penalty, against the 0.52 m/s the smoothing removes in the
    first place. Setting one penalty for the whole cohort is deliberate --
    letting GCV pick per athlete would smooth each person's target differently
    and put a new inconsistency between athletes where none existed.
    """
    v = np.asarray(vel, dtype=float)
    B = _bspline_basis(len(v), n_basis, 3)
    P = _second_difference_penalty(B.shape[1])
    return B @ np.linalg.solve(B.T @ B + penalty * P, B.T @ v)


def peak_velocity_smoothed(vel, penalty=30.0, n_basis=60):
    """Peak of the smoothed velocity curve. Returns (peak, smoothed_curve)."""
    sm = smooth_velocity_curve(vel, penalty, n_basis)
    return float(sm.max()), sm


def velocity_penalty_sweep(feat, penalties=(0.01, 0.1, 1, 10, 100, 1000),
                           n_basis=60):
    """How much does the smoothing penalty actually change the answer?

    A smoothing choice that moves the result is a knob the analyst is turning.
    This reports, across a very wide range of penalties, the effective degrees
    of freedom left in the curve, the mean peak, how much of the raw peak the
    smoothing removed, and the correlation between knee range of motion -- the
    strongest single feature in the table -- and the resulting target. So the
    reader can see whether the conclusion depends on the setting or survives it.
    """
    pids = list(feat["y"].index)
    x    = feat["X"]["knee_hi_ROM"].values
    rows = []
    for lam in penalties:
        fits = [smooth_velocity_curve(feat["vel"][p][0], lam, n_basis)
                for p in pids]
        pk   = np.array([f.max() for f in fits])
        B    = _bspline_basis(len(fits[0]), n_basis, 3)
        P    = _second_difference_penalty(B.shape[1])
        edf  = float(np.trace(B @ np.linalg.solve(B.T @ B + lam * P, B.T)))
        rows.append({"penalty": lam, "edf": round(edf, 1),
                     "mean_peak_ms": round(float(pk.mean()), 3),
                     "noise_removed_ms": round(float((feat["y_raw"].values - pk).mean()), 3),
                     "r_knee_ROM": round(float(np.corrcoef(x, pk)[0, 1]), 3)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Limbs ordered by range of motion, not by anatomical side
# --------------------------------------------------------------------------

PAIRED_JOINTS = ["hip", "knee", "ankle", "shoulder", "elbow"]


def limb_rom_order(cycle):
    """Per joint, report whether the LEFT limb is the larger-ROM one.

    Why this is needed: "right hip" is not the same thing in every athlete.
    Some sprinters drive harder with the left leg, some with the right, and
    averaging a right column across the cohort therefore averages the big side
    of one athlete with the small side of another. That smears exactly the
    variation we are trying to model.

    Why it is decided PER JOINT rather than per person: measured on this
    cohort, all five joints agree on the same side in 0 of 30 athletes, and
    even the leg chain alone (hip, knee, ankle) is unanimous in only 3 of 30.
    Cross-joint agreement runs 0.20-0.77 where chance is 0.50. There is no
    single dominant side to find, so imposing one would be labelling noise.

    Returns {joint: True if left is the larger-ROM limb}.
    """
    out = {}
    for j in PAIRED_JOINTS:
        r = np.ptp(cycle[:, ANGLE_NAMES.index(j + "_R")])
        l = np.ptp(cycle[:, ANGLE_NAMES.index(j + "_L")])
        out[j] = bool(l > r)
    return out


HILO_NAMES = ([f"{j}_{s}" for j in PAIRED_JOINTS for s in ("hi", "lo")]
              + ["trunk_lean"])


def recode_limbs(cycle, order=None):
    """Rewrite a (points, 13) cycle as (points, 11) with HI/LO limb columns.

    Column order is `HILO_NAMES`: hip_hi, hip_lo, knee_hi, knee_lo, ... then
    trunk_lean.

    Two things happen here.

    THE SWAP. For each joint the larger-ROM side becomes `_hi` and the other
    `_lo`. Nothing is discarded and nothing is averaged -- the same numbers come
    out, relabelled.

    THE PHASE ROLL. The cycle is defined right-foot contact to right-foot
    contact, so a right-limb curve starts at contact but a left-limb curve
    starts half a cycle later. If `_hi` were sometimes the raw left column, the
    HI curves would be a mix of contact-referenced and mid-cycle-referenced
    traces, and that phase difference would look like between-athlete variation
    while being nothing of the kind. Rolling the left curve by +50 % of the
    cycle puts it back on the right's phase -- the same correction already used
    in `plot_fast_vs_slow`, where the shift gives r = 1.00 against 0.42 for the
    vertical flip it is often confused with.

    The two SEPARATION columns are dropped. `thigh_separation` is exactly
    `hip_R - hip_L` and `arm_separation` is exactly `shoulder_R - shoulder_L`,
    so the 13-column set has rank 11: they carry no information of their own and
    simply weight hip and shoulder twice. Under HI/LO the same quantity is
    available as `hip_hi - hip_lo` anyway.
    """
    order = limb_rom_order(cycle) if order is None else order
    half  = cycle.shape[0] // 2
    cols  = []
    for j in PAIRED_JOINTS:
        r = cycle[:, ANGLE_NAMES.index(j + "_R")]
        l = np.roll(cycle[:, ANGLE_NAMES.index(j + "_L")], half)
        cols += [l, r] if order[j] else [r, l]
    cols.append(cycle[:, ANGLE_NAMES.index("trunk_lean")])
    return np.column_stack(cols)


# --------------------------------------------------------------------------
# Curves -> scalars
# --------------------------------------------------------------------------

def scalar_features(cycle):
    """Summarise one athlete's cycle as a dict of scalars.

    Range of motion and mean angle for each joint in HI/LO form, plus trunk
    lean.

    These summaries are PHASE-INVARIANT -- a range, a mean and a minimum are
    the same whatever the starting point of the cycle -- so for this block the
    roll inside `recode_limbs` changes nothing. It matters only for the
    functional (fPCA) features, where the shape at each percentage is read.
    """
    hilo = recode_limbs(cycle)
    f = {}
    for k, name in enumerate(HILO_NAMES):
        if name == "trunk_lean":
            continue
        f[f"{name}_ROM"]  = float(np.ptp(hilo[:, k]))
        f[f"{name}_mean"] = float(hilo[:, k].mean())
    f["trunk_lean_mean"] = float(cycle[:, ANGLE_NAMES.index("trunk_lean")].mean())
    f["trunk_lean_ROM"]  = float(np.ptp(cycle[:, ANGLE_NAMES.index("trunk_lean")]))
    return f


def asymmetry_noise_floor(raw64, mk, peak_frame):
    """How much left-right ROM difference would appear in a PERFECTLY symmetric athlete?

    This is the control the HI/LO recoding needs. Taking the larger of two noisy
    paired measurements and calling it HI biases it upward, and the smaller
    downward, so `ROM_hi - ROM_lo` comes out positive even when the true
    asymmetry is zero. Reading that as biomechanics would be reading the
    measurement error.

    The floor is estimated WITHIN an athlete: the same limb measured on
    different strides of the same run should give the same ROM, so the spread
    of that repeated measurement is the noise. Anything smaller than this floor
    is not evidence of asymmetry.

    Returns {joint: {"between_limb": deg, "within_limb": deg}}.
    """
    windows, _ = find_top_speed_strides(mk, peak_frame)
    if len(windows) < 2:
        return {}
    angles, _ = compute_joint_angles(raw64)

    per = []
    for a, b in windows:
        per.append(np.column_stack([resample_to_cycle(angles[a:b + 1, j])
                                    for j in range(angles.shape[1])]))
    per = np.array(per)                                   # (strides, pts, 13)

    out = {}
    for j in PAIRED_JOINTS:
        r = np.ptp(per[:, :, ANGLE_NAMES.index(j + "_R")], axis=1)
        l = np.ptp(per[:, :, ANGLE_NAMES.index(j + "_L")], axis=1)
        out[j] = {
            # the asymmetry we would report, from the stride-averaged curves
            "between_limb": float(abs(r.mean() - l.mean())),
            # the same limb, measured again on another stride: pure noise
            "within_limb":  float(np.mean([r.std(ddof=1), l.std(ddof=1)])),
        }
    return out


def build_feature_table(n_points=101, accel_steps=3, verbose=True):
    """One pass over the cohort producing everything the ML stage needs.

    Returns a dict:
        X        DataFrame, one row per athlete, scalar features
        y        Series,  smoothed peak velocity (m/s) -- the target
        y_raw    Series,  unsmoothed peak, kept so the two can be compared
        height   Series,  measured stature (m)
        cycles   {pid: (points, 13)} raw angle cycles, top speed
        hilo     {pid: (points, 11)} HI/LO recoded cycles
        accel    {pid: (points, 13)} acceleration cycles, where available
        vel      {pid: (raw_curve, smoothed_curve, fs)}
        floors   {pid: asymmetry noise floor}

    Velocity is UNSCALED throughout. The 1.75 m height rescaling elsewhere in
    this module is for drawing only; it never touches the model, because
    scaling an athlete's body also scales the speed derived from it.
    """
    X, y, y_raw, height = {}, {}, {}, {}
    cycles, hilo, accel, vel, floors = {}, {}, {}, {}, {}

    for fpath in sorted(C3D_DIR.glob("*.c3d")):
        pid = fpath.stem.split("-")[0].strip()
        if pid in EXCLUDED_PIDS:
            continue
        try:
            tr = clean_for_angles(fpath)
            cy, _n, _sp = stride_cycle_angles(tr["raw64"], tr["mk"],
                                              tr["peak_frame"], n_points)
            if cy is None:
                continue

            cycles[pid] = cy
            hilo[pid]   = recode_limbs(cy)
            X[pid]      = scalar_features(cy)

            v_raw, _pf, _pv = compute_velocity(tr["mk"], tr["fs"], raw64=tr["raw64"])
            v_peak, v_sm    = peak_velocity_smoothed(v_raw)
            vel[pid]        = (v_raw, v_sm, tr["fs"])
            y[pid]          = v_peak
            y_raw[pid]      = tr["peak_vel_ms"]
            height[pid]     = PARTICIPANT_ANTHRO[pid]["body_height"]
            floors[pid]     = asymmetry_noise_floor(tr["raw64"], tr["mk"],
                                                    tr["peak_frame"])

            steps, _ = find_first_steps(tr["mk"], n_steps=accel_steps)
            if len(steps) == accel_steps:
                ac, _n, _sa = stride_cycle_angles(tr["raw64"], tr["mk"],
                                                  tr["peak_frame"], n_points,
                                                  windows=steps)
                if ac is not None:
                    accel[pid] = ac
        except Exception as e:
            print(f"  skipped {pid}: {e}")

    pids = sorted(X)
    out = {"X":      pd.DataFrame([X[p] for p in pids], index=pids),
           "y":      pd.Series([y[p] for p in pids], index=pids, name="v_peak"),
           "y_raw":  pd.Series([y_raw[p] for p in pids], index=pids, name="v_peak_raw"),
           "height": pd.Series([height[p] for p in pids], index=pids, name="height_m"),
           "cycles": cycles, "hilo": hilo, "accel": accel,
           "vel": vel, "floors": floors}

    if verbose:
        infl = out["y_raw"] - out["y"]
        print(f"{len(pids)} athletes, {out['X'].shape[1]} scalar features")
        print(f"target  : {out['y'].min():.2f}-{out['y'].max():.2f} m/s "
              f"(sd {out['y'].std():.2f})")
        print(f"smoothing removed {infl.mean():+.2f} m/s of noise from the peak "
              f"(range {infl.min():+.2f} to {infl.max():+.2f})")
        print(f"acceleration cycles available for {len(accel)} athletes")
    return out


# ==========================================================================
# 10 · Models and cross-validation
# ==========================================================================
#
# With 30 athletes, a model that looks good on the data it was fitted to tells
# you almost nothing. Everything here is scored by LEAVE-ONE-OUT
# cross-validation: fit on 29, predict the 30th, repeat, and judge only the
# predictions made for athletes the model had never seen.
#
# Two rules are enforced rather than assumed.
#
#   Nothing crosses the fold.  Standardisation, the fPCA basis and the ridge
#                              penalty are all refit on the 29. Fitting any of
#                              them on all 30 first lets the held-out athlete
#                              influence its own prediction, which inflates the
#                              score silently.
#   The null is on the chart.  An intercept-only model scores -0.070 here, not
#                              0.000, because LOO makes even the mean slightly
#                              worse than the full-sample mean. That number is
#                              the honest zero line.

from sklearn.linear_model import LinearRegression, RidgeCV     # noqa: E402


def loo_predict(X, y, model_fn=None, transform=None):
    """Leave-one-out predictions, with all preprocessing refit inside the fold.

    Pass EITHER `X` (a fixed feature matrix) or `transform`, a callable
    `(train_idx, test_idx) -> (X_train, X_test)` for features that must be
    derived from the training athletes only -- fPCA scores being the case that
    matters here.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    model_fn = model_fn or (lambda: LinearRegression())
    Xa = None if X is None else np.asarray(X, dtype=float).reshape(n, -1)

    pred = np.zeros(n)
    for i in range(n):
        tr = np.delete(np.arange(n), i)
        te = np.array([i])
        Xtr, Xte = (transform(tr, te) if transform is not None
                    else (Xa[tr], Xa[te]))
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        m = model_fn()
        m.fit((Xtr - mu) / sd, y[tr])
        pred[i] = float(m.predict((Xte - mu) / sd)[0])
    return pred


def fit_metrics(y, pred, label=""):
    """LOO R-squared, RMSE and MAE for one set of held-out predictions.

    R-squared is 1 - PRESS/TSS: the squared prediction error against the error
    you would make by always guessing the cohort mean. It CAN go negative, and
    a negative value has a plain reading -- the model is worse than the mean.
    """
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    err = y - pred
    return {"model": label,
            "LOO_R2": float(1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "RMSE":   float(np.sqrt((err ** 2).mean())),
            "MAE":    float(np.abs(err).mean())}


def fpca_fold_transform(curves_by_pid, pids, n_comp=6):
    """A `transform` for `loo_predict` that refits fPCA on the training fold.

    The standardisation, the mean curve and the component basis are all
    computed from the training athletes, then the held-out athlete is projected
    onto that basis. Running `run_angle_fpca` on all 30 and cross-validating
    only the regression would let every athlete help build the axes used to
    predict them -- the most common way a small-n result is accidentally faked.
    """
    A = np.stack([curves_by_pid[p] for p in pids])          # (n, pts, ang)

    def _transform(tr, te):
        sd = A[tr].std(axis=(0, 1), keepdims=True)
        sd = np.where(sd < 1e-12, 1.0, sd)
        flat_tr = (A[tr] / sd).reshape(len(tr), -1)
        flat_te = (A[te] / sd).reshape(len(te), -1)
        mean    = flat_tr.mean(axis=0)
        _U, _S, Vt = np.linalg.svd(flat_tr - mean, full_matrices=False)
        V = Vt[:n_comp]
        return (flat_tr - mean) @ V.T, (flat_te - mean) @ V.T

    return _transform


def fpca_scores_leaky(curves_by_pid, pids, n_comp=6):
    """fPCA scores fitted on ALL athletes -- deliberately leaky, for contrast."""
    f = run_angle_fpca({p: curves_by_pid[p] for p in pids}, n_components=n_comp)
    return f["scores"]


def model_ladder(specs, y, verbose=True):
    """Score a list of candidate models, simplest first.

    `specs` is a list of (label, X_or_None, model_fn_or_None, transform_or_None).
    Returns a DataFrame sorted in the order given, so the ladder reads as a
    story rather than a leaderboard.
    """
    rows, preds = [], {}
    for label, X, model_fn, transform in specs:
        p = loo_predict(X, y, model_fn, transform)
        preds[label] = p
        rows.append(fit_metrics(y, p, label))
    table = pd.DataFrame(rows)
    if verbose:
        for _, r in table.iterrows():
            print(f"  {r['model']:<28} LOO R2 = {r['LOO_R2']:+.3f}   "
                  f"RMSE = {r['RMSE']:.3f}   MAE = {r['MAE']:.3f}")
    return table, preds


def permutation_test(X, y, model_fn=None, transform=None, n_perm=5000, seed=0):
    """How often does a shuffled target score as well as the real one?

    The model is rebuilt from scratch on each shuffle, so this asks the right
    question: given this many features, this many athletes and this much
    freedom, how good a LOO R-squared turns up by luck alone?

    Returns (observed_R2, p_value, null_distribution).
    """
    y = np.asarray(y, dtype=float)
    obs = fit_metrics(y, loo_predict(X, y, model_fn, transform))["LOO_R2"]

    rng  = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        yp = rng.permutation(y)
        null[k] = fit_metrics(yp, loo_predict(X, yp, model_fn, transform))["LOO_R2"]

    # +1 in both places: the observed value is itself one of the arrangements.
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return float(obs), p, null


def nested_cv(specs, y, verbose=True):
    """Outer LOO around the CHOICE of model, not just around the fit.

    The ladder reports how the winning model scores. It does not report the
    cost of having picked it -- that choice was informed by all 30 athletes.
    Here the inner loop re-runs the whole ladder on each training fold, picks
    the best candidate there, and only then predicts the held-out athlete.

    The result is normally lower than the ladder's best. That gap is the price
    of selection, and reporting it is the difference between a cross-validated
    model and a cross-validated model-selection procedure.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    pred, chosen = np.zeros(n), []

    for i in range(n):
        tr = np.delete(np.arange(n), i)
        best, best_r2 = None, -np.inf
        for label, X, model_fn, transform in specs:
            Xi = None if X is None else np.asarray(X, float).reshape(n, -1)[tr]
            ti = None
            if transform is not None:
                # re-index the transform onto the training subset
                ti = (lambda t2, e2, _tr=tr, _f=transform:
                      _f(_tr[t2], _tr[e2]))
            r2 = fit_metrics(y[tr], loo_predict(Xi, y[tr], model_fn, ti))["LOO_R2"]
            if r2 > best_r2:
                best, best_r2 = (label, X, model_fn, transform), r2
        label, X, model_fn, transform = best
        chosen.append(label)
        Xa = None if X is None else np.asarray(X, float).reshape(n, -1)
        Xtr, Xte = (transform(tr, np.array([i])) if transform is not None
                    else (Xa[tr], Xa[[i]]))
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        m = (model_fn or (lambda: LinearRegression()))()
        m.fit((Xtr - mu) / sd, y[tr])
        pred[i] = float(m.predict((Xte - mu) / sd)[0])

    if verbose:
        picks = pd.Series(chosen).value_counts()
        print(f"  nested LOO R2 = {fit_metrics(y, pred)['LOO_R2']:+.3f}")
        print("  model chosen by the inner loop:")
        for k, v in picks.items():
            print(f"    {k:<28} {v:2d}/{n} folds")
    return fit_metrics(y, pred, "nested (selection inside fold)"), pred, chosen


def influence_report(X, y, pids):
    """Leverage and Cook's distance for an OLS fit on `X`.

    With 30 athletes a single person can carry a correlation. Cook's distance
    asks, for each athlete, how far the fitted model would move if they were
    dropped. The usual rough flag is D > 4/n, here 0.133.
    """
    X = np.asarray(X, float).reshape(len(y), -1)
    y = np.asarray(y, float)
    A = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    fitted = A @ beta
    resid  = y - fitted
    H      = A @ np.linalg.pinv(A.T @ A) @ A.T
    h      = np.diag(H)
    p      = A.shape[1]
    mse    = (resid ** 2).sum() / max(len(y) - p, 1)
    cooks  = resid ** 2 * h / (p * mse * (1 - h) ** 2 + 1e-12)
    return pd.DataFrame({"pid": pids, "leverage": h, "residual": resid,
                         "cooks_D": cooks}).sort_values("cooks_D",
                                                        ascending=False)


# ==========================================================================
# 11 · Statistical parametric mapping — WHERE in the stride the effect is
# ==========================================================================
#
# A range of motion is one number standing in for a 101-point curve, and it
# cannot say WHEN in the stride fast and slow athletes diverge. Knee ROM mixes
# two mechanically different things -- how far the knee collapses under load in
# stance, and how far it folds in swing -- which have different causes and
# different coaching answers.
#
# The test below correlates velocity against the angle at each phase point and
# then asks which stretches of the curve are longer than chance produces. The
# cluster step is what makes it a test rather than 101 separate ones: a single
# point crossing threshold means little, a long contiguous run means a lot.


def loo_r2_fast(X, y):
    """LOO R-squared for OLS, via the hat matrix rather than by refitting.

    For a linear model the leave-one-out residual has a closed form,
    e_i / (1 - h_ii), so the whole cross-validation costs one fit. This agrees
    with `loo_predict` to six decimal places and is what makes the model-search
    null below affordable -- that null needs tens of thousands of fits, which is
    hours of work the slow way and seconds this way.

    Only valid for plain least squares. Ridge and the fPCA transforms still have
    to go through `loo_predict`, because their preprocessing is refit per fold.
    """
    y = np.asarray(y, float)
    A = np.column_stack([np.ones(len(y)), np.asarray(X, float).reshape(len(y), -1)])
    H = A @ np.linalg.pinv(A.T @ A) @ A.T
    e = y - H @ y
    loo = e / (1 - np.clip(np.diag(H), None, 1 - 1e-12))
    return float(1 - (loo ** 2).sum() / ((y - y.mean()) ** 2).sum())


def _tstat_curve(curves, v):
    """Point-by-point correlation with velocity, as a t statistic.

    `curves` is (athletes, points). Returns (t, r), both length `points`.
    """
    c = curves - curves.mean(axis=0)
    vv = v - v.mean()
    denom = np.sqrt((c ** 2).sum(0) * (vv ** 2).sum())
    r = (c * vv[:, None]).sum(0) / np.where(denom < 1e-12, 1.0, denom)
    r = np.clip(r, -0.999999, 0.999999)
    t = r * np.sqrt(len(v) - 2) / np.sqrt(1 - r ** 2)
    return t, r


def _clusters(t, t_crit):
    """Contiguous runs where |t| exceeds the threshold, with their mass.

    Cluster mass is the summed |t| over the run. Mass rather than length,
    because a long weak run and a short strong one should not score the same.
    """
    over = np.abs(t) >= t_crit
    out, i = [], 0
    while i < len(over):
        if over[i]:
            j = i
            while j + 1 < len(over) and over[j + 1]:
                j += 1
            out.append((i, j, float(np.abs(t[i:j + 1]).sum())))
            i = j + 1
        else:
            i += 1
    return out


def spm_correlation(curves_by_pid, peak_vel, angle, alpha=0.05, n_perm=5000,
                    seed=0):
    """Where in the stride does this angle track speed? Cluster-level test.

    One curve per athlete, one scalar velocity per athlete. At every phase point
    the correlation is turned into a t statistic; runs of points above the
    uncorrected threshold become candidate clusters; and the null is built by
    shuffling the velocity labels and recording the LARGEST cluster mass each
    time. A cluster is significant when it beats that distribution.

    Shuffling the labels is the right null here because it preserves the shape
    and the smoothness of every athlete's curve -- only the pairing with speed
    is destroyed. A pointwise null would ignore that neighbouring points are
    correlated, which is exactly why 101 separate tests over-report.

    Returns a dict with the t curve, the r curve, the clusters and their
    p-values, and the null distribution.
    """
    pids = sorted(curves_by_pid)
    j = ANGLE_NAMES.index(angle)
    C = np.array([curves_by_pid[p][:, j] for p in pids])
    v = np.array([peak_vel[p] for p in pids], dtype=float)

    t_crit = float(stats.t.ppf(1 - alpha / 2, len(v) - 2))
    t, r = _tstat_curve(C, v)
    found = _clusters(t, t_crit)

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        tp, _ = _tstat_curve(C, rng.permutation(v))
        cl = _clusters(tp, t_crit)
        null[i] = max((c[2] for c in cl), default=0.0)

    clusters = [{"start_pct": a, "end_pct": b, "mass": m,
                 "peak_r": float(r[a:b + 1][np.argmax(np.abs(r[a:b + 1]))]),
                 "p": float((null >= m).mean())}
                for a, b, m in found]
    return {"angle": angle, "t": t, "r": r, "t_crit": t_crit,
            "clusters": clusters, "null": null, "n": len(v), "pids": pids}


def spm_report(curves_by_pid, peak_vel, angles, alpha=0.05, n_perm=5000,
               seed=0, verbose=True):
    """`spm_correlation` over a PRE-SPECIFIED list of angles, as a table.

    The list is an argument and not a loop over all thirteen on purpose. With 30
    athletes, testing every angle and reporting the best one reintroduces the
    largest-of-many problem that the cluster test was brought in to control.
    """
    rows, detail = [], {}
    for a in angles:
        res = spm_correlation(curves_by_pid, peak_vel, a, alpha, n_perm, seed)
        detail[a] = res
        if not res["clusters"]:
            rows.append({"angle": a, "cluster": "none", "peak_r": np.nan,
                         "mass": np.nan, "p": np.nan})
        for c in res["clusters"]:
            rows.append({"angle": a,
                         "cluster": f'{c["start_pct"]}-{c["end_pct"]} %',
                         "peak_r": round(c["peak_r"], 3),
                         "mass": round(c["mass"], 1),
                         "p": round(c["p"], 4)})
    table = pd.DataFrame(rows)
    if verbose:
        print(f"SPM, {n_perm} label shuffles, cluster-mass inference "
              f"(alpha = {alpha}, n = {len(peak_vel)}):")
        print(table.to_string(index=False))
    return table, detail


# ==========================================================================
# 12 · Acceleration process model — step mechanics and the v(t) profile
# ==========================================================================
#
# Sections 4-9 describe the SHAPE of the movement (joint-angle cycles, fPCA).
# This section describes the OUTCOME of the first steps out of the blocks: how
# far and how often the athlete steps, and how quickly forward velocity rises.
#
# These features are PRE-SPECIFIED and live in their own table. They are not
# wired into `build_feature_table`'s audited X. Touchdown geometry is deferred.
#
# Two honest limits:
#   * 60 Hz sampling. One frame is 16.7 ms, so a ~100 ms ground contact is
#     resolved to about ±17 ms. Contact and flight times are KINEMATIC estimates
#     from foot height, not force-plate measurements.
#   * Mass-free by design. No force plates and no measured body mass. The
#     analysis target is a0, V0, tau, RFmax, DRF (mass cancels). F0 and Pmax
#     are optional unit conversions if a mass is passed; they are not the
#     scientific default and are never filled with a dummy kg.


_CONTACT_BAND_M = 0.03   # foot is "down" within 3 cm of its height minimum
_G_MS2 = 9.81


def _monoexp_vt(t, vmax, tau):
    return vmax * (1.0 - np.exp(-t / tau))


def _contact_band(foot_z, contact, contact_band_m=_CONTACT_BAND_M):
    """Walk out from a height minimum while the foot stays near the floor."""
    floor = foot_z[contact]
    td = int(contact)
    while td > 0 and foot_z[td - 1] - floor <= contact_band_m:
        td -= 1
    to = int(contact)
    while to < len(foot_z) - 1 and foot_z[to + 1] - floor <= contact_band_m:
        to += 1
    return td, to


def foot_contact_events(mk, foot="both", prominence=STRIDE_PROMINENCE_M):
    """Kinematic touchdown / toe-off from foot-height minima.

    Same peak finder as `find_first_steps` (`ANGLE_STRIDE_MIN_DISTANCE` and
    `prominence`, default `STRIDE_PROMINENCE_M`). Around each minimum the foot
    is taken to be down while its height stays within 3 cm of that trough.
    There is no force plate; at 60 Hz the edges are good to about one frame.

    Parameters
    ----------
    mk : ndarray, shape (n_frames, n_segments, 3)
        Segment centroids after `clean_for_angles` (ISB / pipeline axes).
    foot : {'both', 'R', 'L'}
    prominence : float
        Metres, passed to `find_peaks` on negated foot height.

    Returns
    -------
    list of dict
        Ordered by contact frame: ``foot``, ``contact`` (trough frame),
        ``touchdown``, ``toeoff``.
    """
    wanted = {"R": IDX_R_FOOT, "L": IDX_L_FOOT}
    if foot == "both":
        sides = ("R", "L")
    elif foot in wanted:
        sides = (foot,)
    else:
        raise ValueError("foot must be 'both', 'R', or 'L'")

    found = []
    for side in sides:
        foot_z = mk[:, wanted[side], _VERT]
        contacts, _ = find_peaks(-foot_z, distance=ANGLE_STRIDE_MIN_DISTANCE,
                                 prominence=prominence)
        for c in contacts:
            td, to = _contact_band(foot_z, int(c))
            found.append({"foot": side, "contact": int(c),
                          "touchdown": td, "toeoff": to})
    found.sort(key=lambda e: e["contact"])
    return found


def com_horizontal_velocity(raw64, fs):
    """Fore-aft velocity of a pelvis-centroid CoM proxy (m/s).

    A true centre of mass needs segment masses this study does not have. The
    proxy is the mean of the seven pelvis markers in `raw64` (`MK_PELVIS_ALL`,
    indices 0–6: the Xsens pelvis cluster). That is the same point
    `horizontal_travel_axis` uses for alignment, and the one least tugged about
    by arm and leg swing. It is NOT the 3-D thoracic speed that
    `compute_velocity` reports for the audited peak.

    After `clean_for_angles` the AP axis is forward, so this is the gradient of
    pelvis AP position. Returns a 1-D array of length n_frames.
    """
    pos_fwd = raw64[:, MK_PELVIS_ALL, _AP].mean(axis=1)
    return np.gradient(pos_fwd, 1.0 / fs)


def step_spatiotemporal(mk, raw64, fs, n_steps=8):
    """Per-step length, contact/flight time, frequency and velocity.

    Step windows come from `find_first_steps` (alternate-foot contact to
    contact). Contact and flight times use the kinematic bands from
    `foot_contact_events`. Step length is the fore-aft distance between the
    contacting heels in `raw64` (`MK_R_HEEL` / `MK_L_HEEL`) at the two contact
    frames.

    Columns: ``step``, ``foot``, ``step_length_m``, ``contact_time_s``,
    ``flight_time_s``, ``step_frequency_hz``, ``step_velocity_ms``.

    ``flight_time_s`` can go slightly negative during early double support when
    kinematic toe-off is a frame late; that is left visible, not clipped.
    """
    windows, _rejected = find_first_steps(mk, n_steps=n_steps)
    events = {e["contact"]: e for e in foot_contact_events(mk, foot="both")}
    heel = {"R": MK_R_HEEL, "L": MK_L_HEEL}
    rows = []
    for i, (a, b) in enumerate(windows):
        e0, e1 = events.get(int(a)), events.get(int(b))
        if e0 is None or e1 is None:
            continue
        foot0, foot1 = e0["foot"], e1["foot"]
        step_len = float(raw64[b, heel[foot1], _AP] - raw64[a, heel[foot0], _AP])
        step_time = (b - a) / fs
        contact_time = (e0["toeoff"] - e0["touchdown"]) / fs
        flight_time = (b - e0["toeoff"]) / fs
        rows.append({
            "step": i + 1,
            "foot": foot0,
            "step_length_m": step_len,
            "contact_time_s": contact_time,
            "flight_time_s": flight_time,
            "step_frequency_hz": (1.0 / step_time) if step_time > 0 else np.nan,
            "step_velocity_ms": (step_len / step_time) if step_time > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def step_feature_slopes(step_df, n_steps=8):
    """Ordinary-least-squares slope of each spatiotemporal column vs step number.

    Fits the first `n_steps` rows (or fewer if the table is shorter). Returns a
    dict ``{column: slope}`` in that column's SI unit per step. Empty / too-short
    columns yield NaN.
    """
    cols = ["step_length_m", "contact_time_s", "flight_time_s",
            "step_frequency_hz", "step_velocity_ms"]
    if step_df is None or len(step_df) == 0:
        return {c: np.nan for c in cols}
    use = step_df.iloc[:n_steps]
    x = use["step"].to_numpy(dtype=float) if "step" in use.columns \
        else np.arange(1, len(use) + 1, dtype=float)
    out = {}
    for c in cols:
        y = use[c].to_numpy(dtype=float) if c in use.columns \
            else np.full(len(use), np.nan)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            out[c] = np.nan
        else:
            out[c] = float(np.polyfit(x[mask], y[mask], 1)[0])
    return out


def velocity_time_fit(vel_h, fs, t_window=None):
    """Fit v(t) = vmax * (1 - exp(-t / tau)) to horizontal velocity.

    Most athletes do not plateau in 60 m, so tau and vmax are extrapolations
    beyond the recorded trial, not observed top speed. They summarise the rise
    that *was* captured; they are not a measured maximum.

    Parameters
    ----------
    vel_h : ndarray
        Horizontal (fore-aft) velocity, typically from `com_horizontal_velocity`.
    fs : float
        Sampling rate in Hz.
    t_window : (t0, t1) or None
        Fit window in seconds from trial t = 0. None uses the whole trace.

    Returns
    -------
    dict
        ``vmax``, ``tau``, ``a0`` (= vmax / tau), ``rmse``, ``r2``,
        ``n_samples``, ``fit_ok``. On failure the numeric fields are NaN and
        ``fit_ok`` is False.
    """
    vel_h = np.asarray(vel_h, dtype=float)
    t = np.arange(len(vel_h), dtype=float) / fs
    if t_window is not None:
        t0, t1 = t_window
        keep = (t >= t0) & (t <= t1)
        t, vel_h = t[keep], vel_h[keep]
    mask = np.isfinite(t) & np.isfinite(vel_h)
    t, vel_h = t[mask], vel_h[mask]

    failed = {"vmax": np.nan, "tau": np.nan, "a0": np.nan, "rmse": np.nan,
              "r2": np.nan, "n_samples": int(len(vel_h)), "fit_ok": False}
    if len(vel_h) < 5:
        return failed

    v_guess = float(np.nanmax(vel_h))
    try:
        popt, _ = curve_fit(
            _monoexp_vt, t, vel_h,
            p0=[max(v_guess, 1.0), 0.8],
            bounds=([0.1, 0.05], [20.0, 8.0]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return failed
    vmax, tau = float(popt[0]), float(popt[1])
    if not np.isfinite(vmax) or not np.isfinite(tau) or tau <= 0:
        return failed
    resid = vel_h - _monoexp_vt(t, vmax, tau)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_tot = float(np.sum((vel_h - vel_h.mean()) ** 2))
    r2 = float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else np.nan
    return {"vmax": vmax, "tau": tau, "a0": vmax / tau, "rmse": rmse,
            "r2": r2, "n_samples": int(len(vel_h)), "fit_ok": True}


def fv_profile_samozino(vel_h, fs, body_mass=None, stature=None, t_window=None):
    """Samozino / Morin field force–velocity profile from v(t).

    Always returns the mass-free quantities V0, tau, a0, RFmax, DRF from the
    mono-exponential fit (aero neglected: Fair needs mass). That mass-free
    profile is the intended analysis. F0 and Pmax are filled only when
    `body_mass` is given; otherwise they are None and ``mass_available`` is
    False. Do not pass a dummy mass to invent newtons.

    `stature` is accepted for a future aero term that also needs mass; it is
    unused on the mass-free path.

    RFmax = a0 / sqrt(a0² + g²) (mass cancels). It is a strictly increasing
    function of a0, so a0 and RFmax are not two independent findings. DRF is
    the OLS slope of the modelled RF–v relationship over the fitted samples.
    """
    _ = stature  # aero area needs mass as well; not used in the mass-free path
    fit = velocity_time_fit(vel_h, fs, t_window=t_window)
    out = {
        "V0": np.nan, "tau": np.nan, "a0": np.nan,
        "RFmax": np.nan, "DRF": np.nan,
        "F0": None, "Pmax": None,
        "mass_available": body_mass is not None,
        "fit_ok": False,
    }
    if not fit["fit_ok"]:
        return out

    V0, tau = fit["vmax"], fit["tau"]
    a0 = V0 / tau
    rfmax = a0 / np.sqrt(a0 ** 2 + _G_MS2 ** 2)

    t = np.arange(len(vel_h), dtype=float) / fs
    if t_window is not None:
        t0, t1 = t_window
        t = t[(t >= t0) & (t <= t1)]
    v_m = _monoexp_vt(t, V0, tau)
    a_m = a0 * np.exp(-t / tau)
    if body_mass is not None:
        # Still no aero: k needs Af(mass, stature). Horizontal force is m*a.
        Fh = body_mass * a_m
        Fg = body_mass * _G_MS2
        RF = Fh / np.sqrt(Fh ** 2 + Fg ** 2)
        out["F0"] = float(body_mass * a0)
        out["Pmax"] = float(body_mass * a0 * V0 / 4.0)
    else:
        RF = a_m / np.sqrt(a_m ** 2 + _G_MS2 ** 2)

    mask = np.isfinite(v_m) & np.isfinite(RF)
    if mask.sum() >= 2:
        drf = float(np.polyfit(v_m[mask], RF[mask], 1)[0])
    else:
        drf = np.nan

    out.update({"V0": float(V0), "tau": float(tau), "a0": float(a0),
                "RFmax": float(rfmax), "DRF": drf, "fit_ok": True})
    return out


def build_accel_feature_table(n_steps=8, t_window=None, masses=None,
                              verbose=False):
    """Cohort table of acceleration-process features.

    Separate from `build_feature_table`. Does not mutate audited X. One row
    per athlete: spatiotemporal means and slopes over the first `n_steps`,
    plus the mass-free Samozino profile (`a0`, `V0`, `tau`, `RFmax`, `DRF`).
    `masses` is an optional {pid: kg} dict for optional F0/Pmax conversion;
    the default (no masses) is the intended mass-free table.

    Returns a DataFrame indexed by pid. Athletes that fail to load are skipped.
    """
    masses = masses or {}
    rows = []
    index = []
    for fpath in sorted(C3D_DIR.glob("*.c3d")):
        pid = fpath.stem.split("-")[0].strip()
        if pid in EXCLUDED_PIDS:
            continue
        try:
            tr = clean_for_angles(fpath)
        except Exception as e:
            if verbose:
                print(f"  skipped {pid}: {e}")
            continue
        steps = step_spatiotemporal(tr["mk"], tr["raw64"], tr["fs"],
                                    n_steps=n_steps)
        slopes = step_feature_slopes(steps, n_steps=n_steps)
        vel_h = com_horizontal_velocity(tr["raw64"], tr["fs"])
        fv = fv_profile_samozino(
            vel_h, tr["fs"],
            body_mass=masses.get(pid),
            stature=PARTICIPANT_ANTHRO.get(pid, {}).get("body_height"),
            t_window=t_window,
        )
        row = {
            "n_steps": int(len(steps)),
            "step_length_m_mean": float(steps["step_length_m"].mean())
                if len(steps) else np.nan,
            "contact_time_s_mean": float(steps["contact_time_s"].mean())
                if len(steps) else np.nan,
            "flight_time_s_mean": float(steps["flight_time_s"].mean())
                if len(steps) else np.nan,
            "step_frequency_hz_mean": float(steps["step_frequency_hz"].mean())
                if len(steps) else np.nan,
            "step_velocity_ms_mean": float(steps["step_velocity_ms"].mean())
                if len(steps) else np.nan,
        }
        row.update({f"slope_{k}": v for k, v in slopes.items()})
        row.update({
            "V0": fv["V0"], "tau": fv["tau"], "a0": fv["a0"],
            "RFmax": fv["RFmax"], "DRF": fv["DRF"],
            "F0": fv["F0"], "Pmax": fv["Pmax"],
            "mass_available": fv["mass_available"],
            "fit_ok": fv["fit_ok"],
        })
        rows.append(row)
        index.append(pid)
    return pd.DataFrame(rows, index=index)

