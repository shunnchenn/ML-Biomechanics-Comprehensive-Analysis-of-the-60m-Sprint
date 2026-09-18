"""
sprint_opensim.py — NOVEL / EXPERIMENTAL OpenSim musculoskeletal bridge.

    ⚠ This module is SEPARATE from the validated 60 m sprint pipeline.
      It imports `sprint_pipeline` READ-ONLY (to reuse the C3D loaders,
      cleaning and the marker-based joint angles) and NEVER modifies it.
      Nothing here touches notebooks 01–05, results/, or the manuscript.

What it does
------------
Drives OpenSim 4.5's command-line tool (`opensim-cmd`, from the installed GUI
app — no Python `opensim` bindings, none are installed) to take ONE sprint
trial from raw markers through a full musculoskeletal analysis:

    C3D ─▶ TRC ─▶ Scale ─▶ Inverse Kinematics ─▶ (swing-phase) Inverse Dynamics

and then compares the OpenSim joint angles against the existing pipeline's
marker-based sagittal angles (`sprint_pipeline.compute_joint_angles`).

Design rules (match the parent project's constraints)
-----------------------------------------------------
* Functions only. No work is done at import time (safe to `import`).
* Dependency-free I/O: the .trc / .mot / .sto readers and writers use only
  numpy + pandas (already in the base env). No `opensim` import.
* Everything it generates lives under `opensim/` (models, trc, setup, out,
  logs). It writes nowhere else.

Model & marker choice
----------------------
Model: **gait2392** (Delp 1990) rather than Rajagopal. Rajagopal is extracted
and available (`opensim/_extract/Models/Rajagopal/`) and is the better *running*
model, but gait2392 ships the standard anatomical Delp marker set whose local
landmark coordinates are community-vetted and map almost 1:1 onto the Xsens
virtual anatomical markers (`pRightKneeLatEpicondyle`, `pRightLatMalleolus`,
…). Reusing those vetted locations means the marker registration involves *no
invented local coordinates* — every model marker sits where thousands of
published gait studies put it. gait2392's coupled Walker/Yamaguchi knee is also
exactly the joint definition behind the hypothesised "~16° knee offset", so it
is the right model to *test* that hypothesis. Rajagopal is the documented next
step.

Units / frames
--------------
* Pipeline marker data is in METRES (mm→m handled in `load_c3d`) and is
  gravity-aligned with **Z up**, plus the pipeline's own alignment putting
  **X = forward (running direction), Y = the athlete's LEFT, Z = up** (a
  right-handed frame; see `sprint_pipeline.rotate_horizontal`).
* OpenSim's gait convention is **X = anterior (forward), Y = up, Z = to the
  subject's RIGHT** (right-handed). The exact, documented rotation is therefore

      (X, Y, Z)_opensim = (X, Z, −Y)_pipeline

  i.e. a −90° rotation about the forward axis. `zup_to_yup` applies it. This is
  the ONLY reframing; magnitudes (metres) are untouched.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import OPENSIM_CMD as CONFIGURED_OPENSIM_CMD
from project_paths import OPENSIM_DIR
import sprint_pipeline as sp   # READ-ONLY reuse of loaders / angles / constants


# ==========================================================================
# 1 · Paths & external tool
# ==========================================================================

HERE          = Path(__file__).resolve().parent
MODELS_DIR    = OPENSIM_DIR / "models"
TRC_DIR       = OPENSIM_DIR / "trc"
SETUP_DIR     = OPENSIM_DIR / "setup"
OUT_DIR       = OPENSIM_DIR / "out"
LOG_DIR       = OPENSIM_DIR / "logs"

MODEL_UNSCALED = MODELS_DIR / "gait2392_simbody.osim"

# The bundled CLI from the installed OpenSim 4.5 GUI app. No Python bindings.
OPENSIM_CMD = str(CONFIGURED_OPENSIM_CMD)

# gait2392's own nominal total mass. The anthropometrics sheet has NO mass
# column. ScaleTool <mass> is left at -1 (generic masses kept). ID moments use
# that generic mass and MUST be read as not athlete-specific. See run_swing_id.
NOMINAL_MODEL_MASS_KG = 75.1646
GAIT2392_GENERIC_STATURE_M = 1.80
MASS_UNMEASURED = (
    "FLAG: body mass is not in the anthropometrics sheet. ScaleTool mass=-1 "
    "(generic gait2392 segment masses retained). ID magnitudes are NOT "
    "athlete-specific."
)
GAIT2392_BODIES = [
    "pelvis", "femur_r", "tibia_r", "talus_r", "calcn_r", "toes_r",
    "femur_l", "tibia_l", "talus_l", "calcn_l", "toes_l", "torso",
]

SAMPLING_RATE = sp.SAMPLING_RATE   # 60 Hz


def _dirs():
    """Create the opensim/ sub-folders if missing. Called by the drivers, not
    at import (keeps import side-effect free)."""
    for d in (MODELS_DIR, TRC_DIR, SETUP_DIR, OUT_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    geom = OPENSIM_DIR / "geometry"
    if geom.is_dir():
        link = MODELS_DIR / "Geometry"
        if not link.exists() and not link.is_symlink():
            link.symlink_to(geom, target_is_directory=True)


# ==========================================================================
# 2 · Marker map  (Xsens virtual landmark  ─▶  gait2392 Delp marker)
# ==========================================================================
#
# Each entry: model_marker_name -> (body, (x,y,z) local location in the model
# body frame [m], xsens_source_label, ik_weight, confidence, note).
#
# Locations are taken VERBATIM from gait2392's shipped Scale marker set
# (gait2392_Scale_MarkerSet.xml) — they are not invented here. `confidence` is
# "exact" when the Xsens landmark is anatomically the same point the Delp marker
# names, and "approx"/"uncertain" otherwise (flagged, would need the user's
# Xsens marker-placement protocol to confirm).
#
# Sign convention for Z: gait2392 +Z is to the subject's RIGHT, so right-side
# markers have +Z, left-side −Z.

MARKER_MAP = {
    # ── Pelvis ────────────────────────────────────────────────────────────
    "R.ASIS":       ("pelvis",  ( 0.020,  0.030,  0.128), "pRightASI",              10.0, "exact",     "right ASIS"),
    "L.ASIS":       ("pelvis",  ( 0.020,  0.030, -0.128), "pLeftASI",               10.0, "exact",     "left ASIS"),
    "V.Sacral":     ("pelvis",  (-0.160,  0.040,  0.000), "pSacrum",                10.0, "exact",     "sacrum / posterior pelvis"),
    # ── Femur (thigh) — distal-only bony markers; hip centre pins proximally ─
    "R.Knee.Lat":   ("femur_r", ( 0.000, -0.404,  0.050), "pRightKneeLatEpicondyle", 5.0, "exact",     "lateral femoral epicondyle"),
    "R.Knee.Med":   ("femur_r", ( 0.000, -0.404, -0.050), "pRightKneeMedEpicondyle", 5.0, "exact",     "medial femoral epicondyle"),
    "L.Knee.Lat":   ("femur_l", ( 0.000, -0.404, -0.050), "pLeftKneeLatEpicondyle",  5.0, "exact",     "lateral femoral epicondyle"),
    "L.Knee.Med":   ("femur_l", ( 0.000, -0.404,  0.050), "pLeftKneeMedEpicondyle",  5.0, "exact",     "medial femoral epicondyle"),
    # ── Tibia (shank) ───────────────────────────────────────────────────────
    "R.Ankle.Lat":  ("tibia_r", (-0.005, -0.410,  0.053), "pRightLatMalleolus",      5.0, "exact",     "lateral malleolus"),
    "R.Ankle.Med":  ("tibia_r", ( 0.006, -0.389, -0.038), "pRightMedMalleolus",      5.0, "exact",     "medial malleolus"),
    "L.Ankle.Lat":  ("tibia_l", (-0.005, -0.410, -0.053), "pLeftLatMalleolus",       5.0, "exact",     "lateral malleolus"),
    "L.Ankle.Med":  ("tibia_l", ( 0.006, -0.389,  0.038), "pLeftMedMalleolus",       5.0, "exact",     "medial malleolus"),
    "R.Shank.Front":("tibia_r", ( 0.050, -0.080,  0.000), "pRightTibialTub",         2.0, "approx",    "tibial tuberosity ≈ anterior proximal shank"),
    "L.Shank.Front":("tibia_l", ( 0.050, -0.080,  0.000), "pLeftTibialTub",          2.0, "approx",    "tibial tuberosity ≈ anterior proximal shank"),
    # ── Foot (all on calcaneus in gait2392's set) ──────────────────────────
    "R.Heel":       ("calcn_r", (-0.020,  0.020,  0.000), "pRightHeelFoot",          2.0, "exact",     "posterior calcaneus / heel"),
    "L.Heel":       ("calcn_l", (-0.020,  0.020,  0.000), "pLeftHeelFoot",           2.0, "exact",     "posterior calcaneus / heel"),
    "R.Toe.Tip":    ("calcn_r", ( 0.260,  0.005,  0.000), "pRightToe",               2.0, "exact",     "toe tip"),
    "L.Toe.Tip":    ("calcn_l", ( 0.260,  0.005,  0.000), "pLeftToe",                2.0, "exact",     "toe tip"),
    "R.Midfoot.Lat":("calcn_r", ( 0.100,  0.020,  0.040), "pRightFifthMetatarsal",   1.0, "approx",    "5th metatarsal ≈ lateral midfoot"),
    "L.Midfoot.Lat":("calcn_l", ( 0.100,  0.020, -0.040), "pLeftFifthMetatarsal",    1.0, "approx",    "5th metatarsal ≈ lateral midfoot"),
    "R.Toe.Med":    ("calcn_r", ( 0.190,  0.005, -0.040), "pRightFirstMetatarsal",   1.0, "uncertain", "1st metatarsal ≈ medial forefoot (fixed on calcn, ignores MTP)"),
    "L.Toe.Med":    ("calcn_l", ( 0.190,  0.005,  0.040), "pLeftFirstMetatarsal",    1.0, "uncertain", "1st metatarsal ≈ medial forefoot (fixed on calcn, ignores MTP)"),
    # ── Torso (head+arms are lumped in gait2392; upper-body only lightly used)─
    "Sternum":      ("torso",   ( 0.070,  0.300,  0.000), "pIJ",                     1.0, "uncertain", "Xsens jugular notch (pIJ) vs model mid-sternum marker"),
    "Top.Head":     ("torso",   ( 0.001,  0.657,  0.000), "pTopOfHead",              0.5, "approx",    "top of head; head lumped into torso"),
    "R.Acromium":   ("torso",   (-0.030,  0.440,  0.150), "pRightAcromion",          1.0, "approx",    "right acromion (arm lumped in torso)"),
    "L.Acromium":   ("torso",   (-0.030,  0.440, -0.150), "pLeftAcromion",           1.0, "approx",    "left acromion (arm lumped in torso)"),
}

# Xsens markers deliberately NOT mapped, with the reason (documented, not
# silently dropped). Keyed by Xsens label -> reason.
DROPPED_XSENS = {
    "pHipOrigin":              "Xsens synthetic hip origin; hip centre is set by the model from the pelvis markers.",
    "pRightGreaterTrochanter": "No community-vetted greater-trochanter landmark in the gait2392 set; placing one needs the Xsens protocol. Femur is still fully posed by the hip joint + 2 knee epicondyles.",
    "pLeftGreaterTrochanter":  "See pRightGreaterTrochanter.",
    "pRightPatella":           "gait2392 has no patella body/marker; dropped rather than invent a location.",
    "pLeftPatella":            "See pRightPatella.",
    "pRightCSI":               "Iliac-crest landmark not in the Delp set; pelvis already posed by ASIS×2 + sacrum.",
    "pLeftCSI":                "See pRightCSI.",
    "pRightIschialTub":        "Ischial tuberosity not in the Delp set; redundant for pelvis pose.",
    "pLeftIschialTub":         "See pRightIschialTub.",
    "pRightPivotFoot":         "Redundant foot construction point.",
    "pLeftPivotFoot":          "Redundant foot construction point.",
    "pRightHeelCenter":        "Redundant with pRightHeelFoot.",
    "pLeftHeelCenter":         "Redundant with pLeftHeelFoot.",
    "arms/hands/spine/head-aux":"gait2392 has no arm segments; upper spine, elbows, wrists, hands and auxiliary head markers have no model home. (A full-body model such as Rajagopal is the way to use them.)",
}


def marker_map_table() -> pd.DataFrame:
    """The Xsens→gait2392 mapping as a DataFrame (for the notebook / report)."""
    rows = []
    for name, (body, loc, src, w, conf, note) in MARKER_MAP.items():
        rows.append({"model_marker": name, "body": body, "xsens_source": src,
                     "ik_weight": w, "confidence": conf,
                     "loc_x": loc[0], "loc_y": loc[1], "loc_z": loc[2],
                     "note": note})
    return pd.DataFrame(rows)


def dropped_table() -> pd.DataFrame:
    return pd.DataFrame([{"xsens_marker": k, "reason": v}
                         for k, v in DROPPED_XSENS.items()])


# ==========================================================================
# 3 · Coordinate reframing  (pipeline Z-up  ─▶  OpenSim Y-up)
# ==========================================================================

def zup_to_yup(xyz: np.ndarray) -> np.ndarray:
    """(..., 3) pipeline (X fwd, Y left, Z up) → OpenSim (X fwd, Y up, Z right).

    Exactly (X, Y, Z)_os = (X, Z, −Y)_pipe — a −90° rotation about the forward
    axis. Pure rotation (det +1), so all distances/angles are preserved.
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    return np.stack([x, z, -y], axis=-1)


# ==========================================================================
# 4 · TRC writer / reader  (dependency-free)
# ==========================================================================

def build_marker_frames(aligned_raw64: np.ndarray):
    """Assemble the (n_frames, n_used_markers, 3) OpenSim-frame array.

    `aligned_raw64` is the cleaned, aligned 64-marker array from
    `sprint_pipeline.clean_for_angles` (metres, X fwd / Y left / Z up). Returns
    (model_marker_names, data_yup) where data_yup is in the OpenSim frame.
    Only markers present in MARKER_MAP are exported, named by their MODEL name.
    """
    names, cols = [], []
    # Map Xsens label -> column index in raw64 using the canonical C3D label list.
    label_to_idx = _xsens_label_index()
    for model_name, (body, loc, src, w, conf, note) in MARKER_MAP.items():
        if src not in label_to_idx:
            continue
        idx = label_to_idx[src]
        names.append(model_name)
        cols.append(aligned_raw64[:, idx, :])
    data = np.stack(cols, axis=1)              # (n_frames, n_markers, 3)
    return names, zup_to_yup(data)


_CANONICAL_XSENS_LABELS = [
    'pHipOrigin','pRightASI','pLeftASI','pRightCSI','pLeftCSI','pRightIschialTub',
    'pLeftIschialTub','pSacrum','pL5SpinalProcess','pL3SpinalProcess',
    'pT12SpinalProcess','pIJ','pT4SpinalProcess','pT8SpinalProcess','pPX',
    'pC7SpinalProcess','pTopOfHead','pRightAuricularis','pLeftAuricularis',
    'pBackOfHead','pRightAcromion','pLeftAcromion','pRightArmLatEpicondyle',
    'pRightArmMedEpicondyle','pLeftArmLatEpicondyle','pLeftArmMedEpicondyle',
    'pRightUlnarStyloid','pRightRadialStyloid','pRightOlecranon',
    'pLeftUlnarStyloid','pLeftRadialStyloid','pLeftOlecranon','pRightTopOfHand',
    'pRightPinky','pRightBallHand','pLeftTopOfHand','pLeftPinky','pLeftBallHand',
    'pRightGreaterTrochanter','pRightKneeLatEpicondyle','pRightKneeMedEpicondyle',
    'pRightPatella','pLeftGreaterTrochanter','pLeftKneeLatEpicondyle',
    'pLeftKneeMedEpicondyle','pLeftPatella','pRightTibialTub','pRightLatMalleolus',
    'pRightMedMalleolus','pLeftTibialTub','pLeftLatMalleolus','pLeftMedMalleolus',
    'pRightHeelFoot','pRightFirstMetatarsal','pRightFifthMetatarsal',
    'pRightPivotFoot','pRightHeelCenter','pRightToe','pLeftHeelFoot',
    'pLeftFirstMetatarsal','pLeftFifthMetatarsal','pLeftPivotFoot',
    'pLeftHeelCenter','pLeftToe',
]


def _xsens_label_index():
    return {lbl: i for i, lbl in enumerate(_CANONICAL_XSENS_LABELS)}


def verify_label_order(labels) -> bool:
    """Guard: confirm a trial's C3D labels match the canonical order this module
    assumes. The drivers call this so a re-ordered export would fail loudly
    instead of silently mislabelling markers."""
    return list(labels) == _CANONICAL_XSENS_LABELS


def write_trc(path, marker_names, data_yup, fs=SAMPLING_RATE, start_time=0.0,
              units="m"):
    """Write a valid OpenSim .trc from an (n_frames, n_markers, 3) array."""
    path = Path(path)
    n_frames, n_markers, _ = data_yup.shape
    dt = 1.0 / fs
    lines = []
    lines.append(f"PathFileType\t4\t(X/Y/Z)\t{path.name}")
    lines.append("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
                 "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames")
    lines.append(f"{fs:.1f}\t{fs:.1f}\t{n_frames}\t{n_markers}\t{units}\t"
                 f"{fs:.1f}\t1\t{n_frames}")
    hdr = ["Frame#", "Time"]
    for m in marker_names:
        hdr += [m, "", ""]
    lines.append("\t".join(hdr))
    sub = ["", ""]
    for k in range(n_markers):
        sub += [f"X{k+1}", f"Y{k+1}", f"Z{k+1}"]
    lines.append("\t".join(sub))
    lines.append("")
    for f in range(n_frames):
        t = start_time + f * dt
        row = [str(f + 1), f"{t:.6f}"]
        for m in range(n_markers):
            x, y, z = data_yup[f, m]
            row += [f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"]
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n")
    return path


def read_trc(path):
    """Minimal TRC reader → (time, {marker: (n,3) array})."""
    path = Path(path)
    raw = path.read_text().splitlines()
    marker_names = raw[3].split("\t")[2:]
    marker_names = [m for m in marker_names if m != ""]
    data_lines = [ln for ln in raw[5:] if ln.strip() != ""]
    arr = np.array([[float(v) for v in ln.split("\t") if v != ""]
                    for ln in data_lines])
    time = arr[:, 1]
    xyz = arr[:, 2:].reshape(len(arr), -1, 3)
    return time, {m: xyz[:, i, :] for i, m in enumerate(marker_names)}


# ==========================================================================
# 5 · Storage (.mot / .sto) reader
# ==========================================================================

def read_storage(path) -> pd.DataFrame:
    """Read an OpenSim .mot/.sto into a DataFrame (dependency-free)."""
    path = Path(path)
    lines = path.read_text().splitlines()
    i = 0
    for i, ln in enumerate(lines):
        if ln.strip().lower() == "endheader":
            break
    header_idx = i + 1
    cols = lines[header_idx].split("\t")
    if len(cols) == 1:
        cols = lines[header_idx].split()
    data = []
    for ln in lines[header_idx + 1:]:
        if ln.strip() == "":
            continue
        parts = ln.split("\t")
        if len(parts) == 1:
            parts = ln.split()
        data.append([float(p) for p in parts])
    return pd.DataFrame(data, columns=cols)


# ==========================================================================
# 6 · Setup-XML generators
# ==========================================================================

def _marker_set_xml() -> str:
    """A pruned MarkerSet containing ONLY the mapped markers, at the gait2392
    vetted local locations."""
    objs = []
    for name, (body, loc, src, w, conf, note) in MARKER_MAP.items():
        objs.append(f"""			<Marker name="{name}">
				<socket_parent_frame>/bodyset/{body}</socket_parent_frame>
				<location>{loc[0]:.6f} {loc[1]:.6f} {loc[2]:.6f}</location>
				<fixed>false</fixed>
			</Marker>""")
    return ("""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40500">
	<MarkerSet name="xsens_to_gait2392">
		<objects>
""" + "\n".join(objs) + """
		</objects>
		<groups/>
	</MarkerSet>
</OpenSimDocument>
""")


def _ik_task_xml(indent="\t\t\t\t") -> str:
    tasks = []
    for name, (body, loc, src, w, conf, note) in MARKER_MAP.items():
        tasks.append(f"""{indent}<IKMarkerTask name="{name}">
{indent}	<apply>true</apply>
{indent}	<weight>{w}</weight>
{indent}</IKMarkerTask>""")
    return "\n".join(tasks)


# Within-segment pairs only (a sprint has no static trial; cross-joint pairs
# change with hip/knee angle and would bias scale). Femur length is not
# measured here — no vetted greater-trochanter in the Delp set — so femur
# inherits the uniform stature ScaleSet.
_MEASUREMENTS = [
    ("torso",  [("V.Sacral", "Top.Head")],
     ["torso"]),
    ("pelvis", [("R.ASIS", "L.ASIS")],
     ["pelvis"]),
    ("shank",  [("R.Shank.Front", "R.Ankle.Lat"), ("L.Shank.Front", "L.Ankle.Lat")],
     ["tibia_r", "tibia_l", "talus_r", "talus_l"]),
    ("foot",   [("R.Heel", "R.Toe.Tip"), ("L.Heel", "L.Toe.Tip")],
     ["calcn_r", "calcn_l", "toes_r", "toes_l"]),
]


def _measurement_set_xml() -> str:
    blocks = []
    for mname, pairs, bodies in _MEASUREMENTS:
        pair_xml = "\n".join(
            f"""						<MarkerPair><markers>{a} {b}</markers></MarkerPair>"""
            for a, b in pairs)
        body_xml = "\n".join(
            f"""						<BodyScale name="{b}"><axes>X Y Z</axes></BodyScale>"""
            for b in bodies)
        blocks.append(f"""				<Measurement name="{mname}">
					<apply>true</apply>
					<MarkerPairSet><objects>
{pair_xml}
					</objects><groups/></MarkerPairSet>
					<BodyScaleSet><objects>
{body_xml}
					</objects><groups/></BodyScaleSet>
				</Measurement>""")
    return "\n".join(blocks)


def _stature_scaleset_xml(factor: float) -> str:
    items = []
    for body in GAIT2392_BODIES:
        items.append(
            f"""				<Scale>
					<scales> {factor:.8f} {factor:.8f} {factor:.8f} </scales>
					<segment>{body}</segment>
					<apply>true</apply>
				</Scale>""")
    return "\n".join(items)


def write_scale_setup(setup_path, model_file, marker_set_file, trc_file,
                      out_model, out_scale, out_marker_set, out_static_motion,
                      mass_kg, height_m, meas_t0, meas_t1, place_t0, place_t1):
    """Generate a Scale setup XML (stature ScaleSet + measurements + MarkerPlacer).

    mass_kg is ignored when None / unmeasured: <mass>-1</mass> keeps generic
    segment masses. <height> is measured stature in millimetres (informational).
    """
    if mass_kg is None:
        mass_xml = "-1"
        notes = ("NOVEL/EXPERIMENTAL. MASS UNMEASURED — generic gait2392 masses "
                 "retained. Stature-scaled then marker-pair refined.")
    else:
        mass_xml = f"{float(mass_kg):.4f}"
        notes = "NOVEL/EXPERIMENTAL. Mass supplied by caller (not from anthro sheet)."
    factor = height_m / GAIT2392_GENERIC_STATURE_M
    xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40500">
	<ScaleTool name="sprint_gait2392">
		<mass>{mass_xml}</mass>
		<height>{height_m*1000.0:.1f}</height>
		<age>-1</age>
		<notes>{notes}</notes>
		<GenericModelMaker>
			<model_file>{model_file}</model_file>
			<marker_set_file>{marker_set_file}</marker_set_file>
		</GenericModelMaker>
		<ModelScaler>
			<apply>true</apply>
			<scaling_order> manualScale measurements </scaling_order>
			<MeasurementSet><objects>
{_measurement_set_xml()}
			</objects><groups/></MeasurementSet>
			<ScaleSet><objects>
{_stature_scaleset_xml(factor)}
			</objects><groups/></ScaleSet>
			<marker_file>{trc_file}</marker_file>
			<time_range>{meas_t0:.6f} {meas_t1:.6f}</time_range>
			<preserve_mass_distribution>true</preserve_mass_distribution>
			<output_model_file>{out_model}</output_model_file>
			<output_scale_file>{out_scale}</output_scale_file>
		</ModelScaler>
		<MarkerPlacer>
			<apply>true</apply>
			<IKTaskSet><objects>
{_ik_task_xml()}
			</objects><groups/></IKTaskSet>
			<marker_file>{trc_file}</marker_file>
			<time_range>{place_t0:.6f} {place_t1:.6f}</time_range>
			<output_motion_file>{out_static_motion}</output_motion_file>
			<output_model_file>{out_model}</output_model_file>
			<output_marker_file>{out_marker_set}</output_marker_file>
			<max_marker_movement>-1</max_marker_movement>
		</MarkerPlacer>
	</ScaleTool>
</OpenSimDocument>
"""
    Path(setup_path).write_text(xml)
    return Path(setup_path)


def write_ik_setup(setup_path, model_file, trc_file, out_mot, results_dir,
                   t0, t1):
    xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40500">
	<InverseKinematicsTool name="sprint_ik">
		<results_directory>{results_dir}</results_directory>
		<model_file>{model_file}</model_file>
		<constraint_weight>Inf</constraint_weight>
		<accuracy>1.0000000000000001e-05</accuracy>
		<time_range>{t0:.6f} {t1:.6f}</time_range>
		<output_motion_file>{out_mot}</output_motion_file>
		<report_errors>true</report_errors>
		<IKTaskSet><objects>
{_ik_task_xml()}
		</objects><groups/></IKTaskSet>
		<marker_file>{trc_file}</marker_file>
		<coordinate_file></coordinate_file>
		<report_marker_locations>false</report_marker_locations>
	</InverseKinematicsTool>
</OpenSimDocument>
"""
    Path(setup_path).write_text(xml)
    return Path(setup_path)


def write_id_setup(setup_path, model_file, coordinates_file, out_force,
                   results_dir, t0, t1, lowpass=10.0):
    """Inverse Dynamics with NO external loads (valid for swing-leg moments)."""
    xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40500">
	<InverseDynamicsTool name="sprint_id_swing">
		<results_directory>{results_dir}</results_directory>
		<model_file>{model_file}</model_file>
		<time_range>{t0:.6f} {t1:.6f}</time_range>
		<forces_to_exclude> Muscles </forces_to_exclude>
		<external_loads_file></external_loads_file>
		<coordinates_file>{coordinates_file}</coordinates_file>
		<lowpass_cutoff_frequency_for_coordinates>{lowpass}</lowpass_cutoff_frequency_for_coordinates>
		<output_gen_force_file>{out_force}</output_gen_force_file>
		<joints_to_report_body_forces></joints_to_report_body_forces>
		<output_body_forces_file></output_body_forces_file>
	</InverseDynamicsTool>
</OpenSimDocument>
"""
    Path(setup_path).write_text(xml)
    return Path(setup_path)


# ==========================================================================
# 7 · Runner
# ==========================================================================

def run_tool(setup_path, log_name=None):
    """Invoke `opensim-cmd run-tool <setup>` from the setup file's directory.

    Returns (returncode, log_path). stdout+stderr are captured to opensim/logs.
    The child runs the bundled Simbody/OpenSim libs; if this is called from a
    sandboxed shell it may need full permissions (the parent drives it that way).
    """
    _dirs()
    setup_path = Path(setup_path)
    log_name = log_name or (setup_path.stem + ".log")
    log_path = LOG_DIR / log_name
    env = dict(os.environ)
    home = Path(OPENSIM_CMD).resolve().parent.parent
    lib_dirs = [home / "lib", home / "sdk" / "lib",
                home / "sdk" / "Simbody" / "lib", home / "bin"]
    extra = ":".join(str(p) for p in lib_dirs if p.exists())
    for key in ("DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
        prev = env.get(key, "")
        env[key] = extra if not prev else extra + ":" + prev
    env["OPENSIM_HOME"] = str(home)
    env["PATH"] = str(home / "bin") + ":" + env.get("PATH", "")
    proc = subprocess.run(
        [OPENSIM_CMD, "run-tool", setup_path.name],
        cwd=str(setup_path.parent), env=env,
        capture_output=True, text=True)
    log_path.write_text(
        f"$ opensim-cmd run-tool {setup_path.name}\n(cwd={setup_path.parent})\n"
        f"--- returncode {proc.returncode} ---\n"
        f"===== STDOUT =====\n{proc.stdout}\n===== STDERR =====\n{proc.stderr}\n")
    return proc.returncode, log_path


# ==========================================================================
# 8 · Swing-phase detection (foot off the ground)
# ==========================================================================

def swing_intervals(res, side="R", clearance=0.05):
    """Contiguous frame intervals where one foot is airborne (swing).

    Uses the pipeline's aligned data: the foot-segment vertical height rising
    `clearance` metres above its per-trial minimum marks swing. Returns a list
    of (start_frame, end_frame) inclusive, longest first.
    """
    mk = res["mk"]
    idx = sp.IDX_R_FOOT if side == "R" else sp.IDX_L_FOOT
    z = mk[:, idx, sp._VERT]
    airborne = (z - z.min()) > clearance
    intervals, i, n = [], 0, len(airborne)
    while i < n:
        if airborne[i]:
            j = i
            while j + 1 < n and airborne[j + 1]:
                j += 1
            intervals.append((i, j))
            i = j + 1
        else:
            i += 1
    intervals.sort(key=lambda ab: (ab[1] - ab[0]), reverse=True)
    return intervals


def swing_interval_near_peak(res, side="R", clearance=0.05, min_len=8):
    """The swing interval closest to peak velocity (a clean top-speed swing)."""
    pf = res["peak_frame"]
    cand = [ab for ab in swing_intervals(res, side, clearance)
            if (ab[1] - ab[0]) >= min_len]
    if not cand:
        return None
    cand.sort(key=lambda ab: abs(0.5 * (ab[0] + ab[1]) - pf))
    return cand[0]


# ==========================================================================
# 9 · IK error parsing & angle comparison
# ==========================================================================

def read_ik_errors(results_dir):
    """Read the IK marker-error .sto and summarise RMS/max (metres)."""
    results_dir = Path(results_dir)
    hits = list(results_dir.glob("*marker_errors.sto"))
    if not hits:
        return None
    df = read_storage(hits[0])
    return df


# gait2392 → pipeline column correspondence for the angle comparison.
# The pipeline angle array columns (sprint_pipeline.ANGLE_NAMES) are matched to
# the OpenSim gait2392 coordinate names.
OSIM_VS_PIPE = {
    "hip_R":   "hip_flexion_r",
    "hip_L":   "hip_flexion_l",
    "knee_R":  "knee_angle_r",
    "knee_L":  "knee_angle_l",
    "ankle_R": "ankle_angle_r",
    "ankle_L": "ankle_angle_l",
}


def parse_scale_log(log_path) -> dict:
    """RMS / max marker errors from a ScaleTool log, if OpenSim printed them."""
    text = Path(log_path).read_text() if Path(log_path).exists() else str(log_path)
    out = {}
    alt = re.findall(
        r"marker error:\s*RMS\s*=\s*([0-9.]+),\s*max\s*=\s*([0-9.]+)",
        text, re.I)
    if not alt:
        alt = re.findall(
            r"RMS error\s*=\s*([0-9.]+).*?max(?:imum)? error\s*=\s*([0-9.]+)",
            text, re.I | re.S)
    if alt:
        out["scale_rms_m"] = float(alt[-1][0])
        out["scale_max_m"] = float(alt[-1][1])
    return out


def _unwrap_deg(deg: np.ndarray) -> np.ndarray:
    """Unwrap 360° IK winding, then shift by a multiple of 360 so the mean
    sits in (−180, 180]. Documented; does not change the motion, only the
    branch of the angle."""
    d = np.degrees(np.unwrap(np.radians(np.asarray(deg, dtype=float))))
    return d - 360.0 * np.round(float(np.mean(d)) / 360.0)


def compare_angles(res, ik_mot_path):
    """Compare OpenSim IK angles vs pipeline marker angles on the SAME frames.

    gait2392 knee_angle is more NEGATIVE in flexion (range about −120..+10 deg).
    The pipeline / ISB reports more POSITIVE flexion. Hip flexion and ankle
    dorsiflexion already share the ISB sign. This function does NOT search for
    a flip that maximises r: it applies the known knee definition conversion
    (−1) and records it. Raw (unconverted) OpenSim series stay in the table.
    """
    ik = read_storage(ik_mot_path)
    ang, names = sp.compute_joint_angles(res["raw64"])       # degrees
    pipe = pd.DataFrame(ang, columns=names)

    n = min(len(ik), len(pipe))
    rows = []
    for pname, oname in OSIM_VS_PIPE.items():
        if oname not in ik.columns:
            continue
        o_raw = _unwrap_deg(ik[oname].to_numpy()[:n])
        p = pipe[pname].to_numpy()[:n]
        is_knee = pname.startswith("knee_")
        conversion = "osim_flex_neg -> ISB_flex_pos (* -1)" if is_knee else "none"
        o_isb = -o_raw if is_knee else o_raw
        r_raw = float(np.corrcoef(o_raw, p)[0, 1])
        r_isb = float(np.corrcoef(o_isb, p)[0, 1])
        rows.append({
            "joint": pname,
            "osim_coord": oname,
            "sign_conversion": conversion,
            "pearson_r_raw": round(r_raw, 4),
            "pearson_r_after_known_conversion": round(r_isb, 4),
            "offset_deg_raw_mean": round(float(np.mean(o_raw - p)), 2),
            "offset_deg_converted_mean": round(float(np.mean(o_isb - p)), 2),
            "peak_offset_deg_converted": round(float(np.max(o_isb) - np.max(p)), 2),
            "osim_native_mean_deg": round(float(o_raw.mean()), 2),
            "osim_converted_mean_deg": round(float(o_isb.mean()), 2),
            "pipe_mean_deg": round(float(p.mean()), 2),
            "osim_converted_peak_deg": round(float(np.max(o_isb)), 2),
            "pipe_peak_deg": round(float(np.max(p)), 2),
            "osim_ROM_deg": round(float(np.ptp(o_isb)), 2),
            "pipe_ROM_deg": round(float(np.ptp(p)), 2),
        })
    return pd.DataFrame(rows), ik, pipe


# ==========================================================================
# 10 · High-level drivers  (one trial, end-to-end)
# ==========================================================================

def export_trc(trial="SB25", clearance=None):
    """clean_for_angles(trial) → write opensim/trc/<pid>.trc. Returns res+paths.

    `res` is the pipeline cleaning result (mk, raw64, fs, peak_frame, …) so the
    later comparison runs on exactly the same frames.
    """
    _dirs()
    res = sp.clean_for_angles(trial)
    # Guard: the marker map assumes a fixed C3D label order.
    import ezc3d
    c = ezc3d.c3d(str(res["fpath"]))
    labels = c["parameters"]["POINT"]["LABELS"]["value"]
    if not verify_label_order(labels):
        raise ValueError(f"C3D label order for {res['pid']} does not match the "
                         f"canonical Xsens order this module assumes.")
    names, data = build_marker_frames(res["raw64"])
    trc = TRC_DIR / f"{res['pid']}.trc"
    write_trc(trc, names, data, fs=res["fs"])
    res["trc"] = trc
    res["marker_names"] = names
    return res


def static_window(res, dur_s=0.15):
    """A short low-motion window (start,end in seconds) for marker placement.

    A sprint has no static pose, so we pick the lowest full-body marker-speed
    window and use a few frames there. Documented limitation: the pose is a
    running-flight pose, not a neutral stand.
    """
    raw = res["raw64"]
    fs = res["fs"]
    speed = np.linalg.norm(np.gradient(raw, 1.0 / fs, axis=0), axis=2).mean(axis=1)
    w = max(2, int(round(dur_s * fs)))
    # rolling mean of speed, pick min-speed window centre
    csum = np.cumsum(np.insert(speed, 0, 0))
    win = (csum[w:] - csum[:-w]) / w
    c = int(np.argmin(win))
    t0 = c / fs
    t1 = (c + w) / fs
    return t0, t1, c


def scale_trial(res, mass_kg=None, static_dur_s=0.15):
    """Write + run the Scale tool. Returns dict of paths / status / mass flag."""
    _dirs()
    pid = res["pid"]
    height_m = float(sp.PARTICIPANT_ANTHRO[pid]["body_height"])
    mass_is_assumed = mass_kg is None
    mass_used = None if mass_is_assumed else float(mass_kg)

    # Relative paths (setup runs with cwd = SETUP_DIR).
    rel_model  = os.path.relpath(MODEL_UNSCALED, SETUP_DIR)
    trc_rel    = os.path.relpath(res["trc"], SETUP_DIR)
    mset_path  = SETUP_DIR / f"{pid}_markerset.xml"
    Path(mset_path).write_text(_marker_set_xml())

    out_model  = os.path.relpath(OUT_DIR / f"{pid}_scaled.osim", SETUP_DIR)
    out_scale  = os.path.relpath(OUT_DIR / f"{pid}_scaleSet.xml", SETUP_DIR)
    out_mset   = os.path.relpath(OUT_DIR / f"{pid}_markers_placed.xml", SETUP_DIR)
    out_static = os.path.relpath(OUT_DIR / f"{pid}_static_ik.mot", SETUP_DIR)

    # measurements averaged over the whole trial; marker placement on a short
    # low-motion window.
    t_end = (res["n_frames"] - 1) / res["fs"]
    pt0, pt1, _ = static_window(res, static_dur_s)

    setup = SETUP_DIR / f"{pid}_scale_setup.xml"
    write_scale_setup(setup, rel_model, mset_path.name, trc_rel,
                      out_model, out_scale, out_mset, out_static,
                      mass_used, height_m, 0.0, t_end, pt0, pt1)
    rc, log = run_tool(setup, f"{pid}_scale.log")
    scale_errs = parse_scale_log(log)
    return {"pid": pid, "returncode": rc, "log": log,
            "scaled_model": OUT_DIR / f"{pid}_scaled.osim",
            "scale_set": OUT_DIR / f"{pid}_scaleSet.xml",
            "height_m": height_m,
            "stature_scale": height_m / GAIT2392_GENERIC_STATURE_M,
            "mass_kg": mass_used,
            "mass_is_assumed": mass_is_assumed,
            "mass_flag": MASS_UNMEASURED if mass_is_assumed else None,
            "generic_mass_kg": NOMINAL_MODEL_MASS_KG,
            "placement_window_s": (pt0, pt1),
            **scale_errs}


def run_ik(res, scaled_model):
    """Write + run IK over the whole trial. Returns paths + error summary."""
    _dirs()
    pid = res["pid"]
    results_dir = os.path.relpath(OUT_DIR, SETUP_DIR)
    model_rel   = os.path.relpath(scaled_model, SETUP_DIR)
    trc_rel     = os.path.relpath(res["trc"], SETUP_DIR)
    mot_path    = OUT_DIR / f"{pid}_ik.mot"
    out_mot     = os.path.relpath(mot_path, SETUP_DIR)
    t_end = (res["n_frames"] - 1) / res["fs"]
    setup = SETUP_DIR / f"{pid}_ik_setup.xml"
    write_ik_setup(setup, model_rel, trc_rel, out_mot, results_dir, 0.0, t_end)
    rc, log = run_tool(setup, f"{pid}_ik.log")
    mot = mot_path if mot_path.exists() else SETUP_DIR / f"{pid}_ik.mot"
    errs = read_ik_errors(OUT_DIR)
    summary = None
    if errs is not None and "marker_error_RMS" in errs.columns:
        summary = {
            "rms_mean_m":  float(errs["marker_error_RMS"].mean()),
            "rms_max_m":   float(errs["marker_error_RMS"].max()),
            "max_mean_m":  float(errs["marker_error_max"].mean()),
            "max_max_m":   float(errs["marker_error_max"].max()),
            "n_frames":    int(len(errs)),
        }
    return {"pid": pid, "returncode": rc, "log": log, "ik_mot": mot,
            "errors": errs, "error_summary": summary}


def run_swing_id(res, scaled_model, ik_mot, side="R", lowpass=10.0,
                 pad_frames=1):
    """Run Inverse Dynamics over ONE swing interval (no GRF). Swing-leg joint
    moments only are valid; everything else (pelvis residuals, stance leg) is
    NOT and must not be reported as ground-truth kinetics."""
    _dirs()
    pid = res["pid"]
    sw = swing_interval_near_peak(res, side=side)
    if sw is None:
        return {"pid": pid, "status": "no swing interval found"}
    a, b = sw
    a = min(a + pad_frames, b)      # trim edges to stay safely airborne
    b = max(b - pad_frames, a)
    fs = res["fs"]
    t0, t1 = a / fs, b / fs

    results_dir = os.path.relpath(OUT_DIR, SETUP_DIR)
    model_rel   = os.path.relpath(scaled_model, SETUP_DIR)
    ik_rel      = os.path.relpath(ik_mot, SETUP_DIR)
    out_force   = f"{pid}_id_swing_{side}.sto"
    setup = SETUP_DIR / f"{pid}_id_swing_{side}_setup.xml"
    write_id_setup(setup, model_rel, ik_rel, out_force, results_dir,
                   t0, t1, lowpass=lowpass)
    rc, log = run_tool(setup, f"{pid}_id_swing_{side}.log")

    sto = OUT_DIR / out_force
    moments = None
    if sto.exists():
        df = read_storage(sto)
        suff = side.lower()
        keep = {"time": "time"}
        for c in df.columns:
            cl = c.lower()
            if cl in (f"hip_flexion_{suff}_moment", f"knee_angle_{suff}_moment",
                      f"ankle_angle_{suff}_moment"):
                keep[c] = c
        moments = df[list(keep)].copy()
    return {"pid": pid, "returncode": rc, "log": log, "id_sto": sto,
            "swing_frames": (a, b), "swing_time_s": (t0, t1),
            "side": side, "lowpass_hz": lowpass, "moments": moments}


def run_all(trial="SB25", mass_kg=None, id_side="R"):
    """Full end-to-end for one trial. Returns a dict of every stage's result."""
    res   = export_trc(trial)
    scl   = scale_trial(res, mass_kg=mass_kg)
    out   = {"res": res, "scale": scl}
    map_df = marker_map_table()
    drop_df = dropped_table()
    map_df.to_csv(OUT_DIR / f"{res['pid']}_marker_map.csv", index=False)
    drop_df.to_csv(OUT_DIR / f"{res['pid']}_dropped_markers.csv", index=False)
    out["marker_map"] = map_df
    out["dropped"] = drop_df
    if scl["returncode"] != 0 or not scl["scaled_model"].exists():
        out["status"] = "scale failed"
        return out
    ik    = run_ik(res, scl["scaled_model"])
    out["ik"] = ik
    if ik["ik_mot"].exists():
        cmp_df, ik_df, pipe_df = compare_angles(res, ik["ik_mot"])
        out["comparison"] = cmp_df
        out["ik_df"] = ik_df
        out["pipe_df"] = pipe_df
        cmp_df.to_csv(OUT_DIR / f"{res['pid']}_angle_comparison.csv", index=False)
        out["id"] = run_swing_id(res, scl["scaled_model"], ik["ik_mot"],
                                 side=id_side)
        if out["id"].get("moments") is not None:
            out["id"]["moments"].to_csv(
                OUT_DIR / f"{res['pid']}_id_swing_{id_side}_moments.csv",
                index=False)
    out["status"] = "ok"
    return out
