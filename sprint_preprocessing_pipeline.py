"""
Sprint Preprocessing Pipeline for XSens MVN Data
=================================================
Implements the Velluci & Beaudette (2023) methodology for analyzing
sprint kinematics from XSens MVN full-body motion capture data.

Pipeline:
    1. Load marker data (23 segments × XYZ × frames)
    2. Auto-detect sprint start
    3. Shift origin so right ankle at first frame = (0,0,0)
    4. PCA-based global coordinate system alignment
    5. Trim to first 60 meters (T8 marker)
    6. Compute instantaneous velocity, find peak
    7. Detect 5 strides centered around peak velocity
    8. Time-normalize each stride to 101 frames (linear interpolation)
    9. Remove forward translation (subtract T12)
   10. Average 5 strides into representative stride
   11. Scale to anthropometrics (height normalization)
   12. Reshape into vector: 23 segments × 3 axes × 101 frames = 6,969 features
   13. Store per-participant row in final matrix

Output:
    processed_data/participant_vectors.npy  — (N, 6969) matrix
    processed_data/metadata.csv             — participant info
    processed_data/logs.txt                 — processing log

Author: Sprint Biomechanics Lab
Methodology: Velluci & Beaudette (2023)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
import logging
import sys
import traceback
from datetime import datetime

# Try importing ezc3d (optional — only needed for .c3d files)
try:
    import ezc3d
    HAS_EZC3D = True
except ImportError:
    HAS_EZC3D = False

# =====================================================================
# CONSTANTS
# =====================================================================

SAMPLING_RATE = 60  # Hz
TN_POINTS = 101     # time-normalized frame count
N_SEGMENTS = 23
N_STRIDES_TARGET = 5
TRIM_DISTANCE_M = 60.0
STRIDE_MIN_DISTANCE = 30        # minimum frames between foot contacts
STRIDE_PROMINENCE_M = 0.02      # prominence for foot contact detection (meters)
SPRINT_START_VEL_THRESH = 1.0   # m/s threshold for sprint start detection
SPRINT_START_SUSTAIN = 5        # frames velocity must be sustained

SEGMENTS = [
    'Pelvis', 'L5', 'L3', 'T12', 'T8', 'Neck', 'Head',
    'Right Shoulder', 'Right Upper Arm', 'Right Forearm', 'Right Hand',
    'Left Shoulder', 'Left Upper Arm', 'Left Forearm', 'Left Hand',
    'Right Upper Leg', 'Right Lower Leg', 'Right Foot', 'Right Toe',
    'Left Upper Leg', 'Left Lower Leg', 'Left Foot', 'Left Toe'
]

# Segment indices (0-based)
IDX_PELVIS = 0
IDX_L5 = 1
IDX_L3 = 2
IDX_T12 = 3
IDX_T8 = 4
IDX_NECK = 5
IDX_HEAD = 6
IDX_R_SHOULDER = 7
IDX_L_SHOULDER = 11
IDX_R_UPPER_LEG = 15
IDX_R_LOWER_LEG = 16
IDX_R_FOOT = 17
IDX_R_TOE = 18
IDX_L_FOOT = 21
IDX_L_TOE = 22

# Thorax segment indices for velocity calculation
THORAX_INDICES = [IDX_L5, IDX_L3, IDX_T12, IDX_T8, IDX_NECK, IDX_R_SHOULDER, IDX_L_SHOULDER]


# =====================================================================
# LOGGING SETUP
# =====================================================================

def setup_logging(output_dir):
    """Configure logging to both console and file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / 'logs.txt'

    logger = logging.getLogger('sprint_pipeline')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# =====================================================================
# PIPELINE FUNCTIONS
# =====================================================================

def find_trial_files(root_dir):
    """
    Recursively scan directories for valid trial files.

    Priority: .c3d > .xlsx
    Groups files by participant ID, keeping highest-priority format.

    Parameters
    ----------
    root_dir : str or Path
        Root directory to scan.

    Returns
    -------
    list[dict]
        Each dict has keys: participant_id, filepath, filetype
    """
    root = Path(root_dir)
    logger = logging.getLogger('sprint_pipeline')

    # Collect all candidate files
    # If ezc3d is available, prefer .c3d; otherwise prefer .xlsx
    candidates = {}
    if HAS_EZC3D:
        priority = {'.c3d': 0, '.xlsx': 1}
    else:
        priority = {'.xlsx': 0, '.c3d': 1}
        logger.info("ezc3d not installed — preferring .xlsx files over .c3d")

    for ext in ['.c3d', '.xlsx']:
        for filepath in root.rglob(f'*{ext}'):
            # Parse participant ID from filename: SB60-001.c3d -> SB60
            stem = filepath.stem
            # Remove trial suffix like -001
            parts = stem.split('-')
            participant_id = parts[0].strip()

            if not participant_id.startswith('SB'):
                continue

            file_priority = priority[ext]
            if participant_id not in candidates or file_priority < candidates[participant_id]['priority']:
                candidates[participant_id] = {
                    'participant_id': participant_id,
                    'filepath': str(filepath),
                    'filetype': ext,
                    'priority': file_priority
                }

    # Convert to list, drop priority field
    trials = []
    for pid in sorted(candidates.keys()):
        entry = candidates[pid]
        trials.append({
            'participant_id': entry['participant_id'],
            'filepath': entry['filepath'],
            'filetype': entry['filetype']
        })

    logger.info(f"Found {len(trials)} trial files:")
    for t in trials:
        logger.info(f"  {t['participant_id']}: {t['filetype']} -> {t['filepath']}")

    return trials


def load_marker_data(filepath, filetype):
    """
    Load marker/segment position data from a trial file.

    Parameters
    ----------
    filepath : str
        Path to the data file.
    filetype : str
        File extension ('.c3d' or '.xlsx').

    Returns
    -------
    marker_data : np.ndarray
        Shape (n_frames, 23, 3) in meters.
    metadata : dict
        Contains 'n_frames', 'sampling_rate', 'source'.
    """
    logger = logging.getLogger('sprint_pipeline')

    if filetype == '.c3d':
        if not HAS_EZC3D:
            raise ImportError("ezc3d is not installed. Install with: pip install ezc3d")

        c = ezc3d.c3d(filepath)
        points = c['data']['points']  # shape: (4, n_markers, n_frames) — [x,y,z,1]
        n_markers_found = points.shape[1]
        n_frames = points.shape[2]

        logger.info(f"C3D loaded: {n_markers_found} markers, {n_frames} frames")

        # Extract XYZ, transpose to (frames, markers, 3)
        marker_data = points[:3, :, :].transpose(2, 1, 0)  # (frames, markers, 3)

        # C3D from XSens may be in mm — check scale
        # If max absolute value > 100, likely in mm
        max_val = np.nanmax(np.abs(marker_data))
        if max_val > 100:
            logger.info(f"C3D data appears to be in mm (max={max_val:.1f}), converting to meters")
            marker_data = marker_data / 1000.0

        # Handle case where C3D has more/fewer markers than 23
        if n_markers_found > N_SEGMENTS:
            logger.warning(f"C3D has {n_markers_found} markers, using first {N_SEGMENTS}")
            marker_data = marker_data[:, :N_SEGMENTS, :]
        elif n_markers_found < N_SEGMENTS:
            logger.warning(f"C3D has only {n_markers_found} markers, expected {N_SEGMENTS}")
            padded = np.zeros((n_frames, N_SEGMENTS, 3))
            padded[:, :n_markers_found, :] = marker_data
            marker_data = padded

        fs = c['header']['points']['frame_rate']
        metadata = {'n_frames': n_frames, 'sampling_rate': fs, 'source': 'c3d'}

    elif filetype == '.xlsx':
        df = pd.read_excel(filepath, sheet_name='Segment Position')
        n_frames = len(df)

        marker_data = np.zeros((n_frames, N_SEGMENTS, 3))
        for i, segment in enumerate(SEGMENTS):
            for j, axis in enumerate(['x', 'y', 'z']):
                col_name = f'{segment} {axis}'
                if col_name in df.columns:
                    marker_data[:, i, j] = df[col_name].values
                else:
                    # Try flexible matching
                    matched = [c for c in df.columns if segment.lower() in c.lower() and axis in c.lower()]
                    if matched:
                        marker_data[:, i, j] = df[matched[0]].values
                    else:
                        logger.warning(f"Column not found: {col_name}")

        metadata = {'n_frames': n_frames, 'sampling_rate': SAMPLING_RATE, 'source': 'xlsx'}

        # Load XSens pre-computed Segment Velocity (less smoothed than position derivative)
        try:
            vel_df = pd.read_excel(filepath, sheet_name='Segment Velocity')
            velocity_data = np.zeros((n_frames, N_SEGMENTS, 3))
            for i, segment in enumerate(SEGMENTS):
                for j, axis in enumerate(['x', 'y', 'z']):
                    col_name = f'{segment} {axis}'
                    if col_name in vel_df.columns:
                        velocity_data[:, i, j] = vel_df[col_name].values
            metadata['velocity_data'] = velocity_data
            logger.info("Loaded XSens Segment Velocity sheet")
        except Exception as e:
            logger.warning(f"Could not load Segment Velocity sheet: {e}")
            metadata['velocity_data'] = None
    else:
        raise ValueError(f"Unsupported file type: {filetype}")

    logger.info(f"Loaded marker data: shape={marker_data.shape}, units=meters")
    return marker_data, metadata


def detect_sprint_start(marker_data, fs=SAMPLING_RATE):
    """
    Automatically detect sprint start frame.

    Uses pelvis forward velocity exceeding threshold for sustained frames.

    Parameters
    ----------
    marker_data : np.ndarray
        Shape (n_frames, 23, 3).
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    int
        Frame index where sprint starts.
    """
    logger = logging.getLogger('sprint_pipeline')

    # Use pelvis (most stable segment) x-position for velocity
    pelvis_x = marker_data[:, IDX_PELVIS, 0]

    # Compute velocity via finite differences
    velocity = np.gradient(pelvis_x, 1.0 / fs)

    # Find first frame where velocity exceeds threshold for SPRINT_START_SUSTAIN consecutive frames
    above_thresh = velocity > SPRINT_START_VEL_THRESH
    start_frame = 0

    for i in range(len(above_thresh) - SPRINT_START_SUSTAIN):
        if all(above_thresh[i:i + SPRINT_START_SUSTAIN]):
            start_frame = max(0, i - 5)  # back up 5 frames to capture onset
            break

    logger.info(f"Sprint start detected at frame {start_frame} "
                f"(velocity={velocity[start_frame]:.2f} m/s)")
    return start_frame


def shift_origin(marker_data):
    """
    Shift origin so the most posterior foot at frame 0 = (0,0,0).

    Parameters
    ----------
    marker_data : np.ndarray
        Shape (n_frames, 23, 3).

    Returns
    -------
    np.ndarray
        Origin-shifted marker data, same shape.
    """
    logger = logging.getLogger('sprint_pipeline')

    # Find which foot is more posterior (smaller x) at frame 0
    r_foot_x = marker_data[0, IDX_R_FOOT, 0]
    l_foot_x = marker_data[0, IDX_L_FOOT, 0]

    if r_foot_x <= l_foot_x:
        origin = marker_data[0, IDX_R_FOOT, :].copy()
        logger.info(f"Origin set to Right Foot at frame 0: {origin}")
    else:
        origin = marker_data[0, IDX_L_FOOT, :].copy()
        logger.info(f"Origin set to Left Foot at frame 0: {origin}")

    # Subtract origin from all frames, all markers
    shifted = marker_data - origin[np.newaxis, np.newaxis, :]
    return shifted


def perform_pca_alignment(marker_data):
    """
    PCA-based global coordinate system realignment.

    PC1 = anteroposterior (sprint direction)
    PC2 = mediolateral
    PC3 = vertical

    Parameters
    ----------
    marker_data : np.ndarray
        Shape (n_frames, 23, 3).

    Returns
    -------
    aligned_data : np.ndarray
        Same shape, rotated into PCA coordinate system.
    pca : PCA
        Fitted PCA object.
    """
    logger = logging.getLogger('sprint_pipeline')
    n_frames, n_markers, _ = marker_data.shape

    # Reshape: all marker positions across all frames into (N, 3)
    reshaped = marker_data.reshape(-1, 3)

    # Remove NaN/inf
    valid_mask = np.isfinite(reshaped).all(axis=1)
    clean_data = reshaped[valid_mask]

    # Fit PCA
    pca = PCA(n_components=3)
    pca.fit(clean_data)

    rotation = pca.components_  # (3, 3) — rows are principal directions

    logger.info(f"PCA explained variance: {pca.explained_variance_ratio_}")

    # Ensure right-handed coordinate system
    if np.linalg.det(rotation) < 0:
        rotation[2, :] *= -1
        logger.info("Flipped PC3 to ensure right-handed coordinate system")

    # Apply rotation to all frames
    aligned = np.zeros_like(marker_data)
    for frame in range(n_frames):
        aligned[frame, :, :] = marker_data[frame, :, :] @ rotation.T

    # Ensure PC1 (x) is positive in sprint direction:
    # T8 should progress positively over time
    t8_x_start = aligned[0, IDX_T8, 0]
    t8_x_end = aligned[-1, IDX_T8, 0]
    if t8_x_end < t8_x_start:
        aligned[:, :, 0] *= -1
        rotation[0, :] *= -1
        logger.info("Flipped PC1 to ensure forward sprint direction is positive")

    # Ensure PC3 (z) is vertical (positive = up):
    # Head should be above pelvis on average
    head_z = np.mean(aligned[:, IDX_HEAD, 2])
    pelvis_z = np.mean(aligned[:, IDX_PELVIS, 2])
    if head_z < pelvis_z:
        aligned[:, :, 2] *= -1
        rotation[2, :] *= -1
        logger.info("Flipped PC3 to ensure vertical axis points upward")

    return aligned, pca, rotation


def trim_to_60m(marker_data):
    """
    Trim data to first 60 meters using T8 marker x-position.

    Parameters
    ----------
    marker_data : np.ndarray
        Shape (n_frames, 23, 3) in PCA-aligned coordinates.

    Returns
    -------
    np.ndarray
        Trimmed marker data.
    """
    logger = logging.getLogger('sprint_pipeline')

    t8_x = marker_data[:, IDX_T8, 0]
    max_distance = np.max(t8_x)

    if max_distance >= TRIM_DISTANCE_M:
        crop_indices = np.where(t8_x >= TRIM_DISTANCE_M)[0]
        crop_frame = crop_indices[0]
        trimmed = marker_data[:crop_frame, :, :]
        logger.info(f"Trimmed to 60m at frame {crop_frame} ({trimmed.shape[0]} frames)")
    else:
        trimmed = marker_data
        logger.warning(f"Max T8 distance = {max_distance:.2f}m (< 60m), using all data")

    return trimmed


def compute_velocity(marker_data, fs=SAMPLING_RATE, velocity_data=None, pca_rotation=None):
    """
    Compute instantaneous velocity from thorax segments.

    Uses XSens pre-computed Segment Velocity when available (more accurate
    than differentiating position, which is over-smoothed by Kalman filtering).
    Falls back to position derivative when velocity data is not available.

    Parameters
    ----------
    marker_data : np.ndarray
        Shape (n_frames, 23, 3) in PCA-aligned coordinates.
    fs : int
        Sampling rate in Hz.
    velocity_data : np.ndarray or None
        XSens Segment Velocity data, shape (n_frames, 23, 3) in raw coordinates.
    pca_rotation : np.ndarray or None
        PCA rotation matrix (3, 3) to align velocity vectors.

    Returns
    -------
    velocity : np.ndarray
        Velocity in m/s, length n_frames.
    peak_vel_frame : int
        Frame index of peak velocity.
    peak_velocity : float
        Peak velocity in m/s.
    """
    logger = logging.getLogger('sprint_pipeline')

    if velocity_data is not None and pca_rotation is not None:
        # Use XSens pre-computed velocity (less smoothed, more accurate peaks)
        # Rotate velocity vectors into PCA coordinate system
        n_frames = velocity_data.shape[0]
        aligned_vel = np.zeros_like(velocity_data)
        for frame in range(n_frames):
            aligned_vel[frame, :, :] = velocity_data[frame, :, :] @ pca_rotation.T

        # Mean thorax velocity along PC1 (sprint direction)
        thorax_vel = np.mean(aligned_vel[:, THORAX_INDICES, :], axis=1)  # (frames, 3)
        velocity = thorax_vel[:, 0]  # PC1 component = forward velocity

        logger.info("Using XSens Segment Velocity (sensor-fused, less smoothed)")
    else:
        # Fallback: differentiate position data
        thorax_pos = np.mean(marker_data[:, THORAX_INDICES, :], axis=1)
        thorax_pc1 = thorax_pos[:, 0]
        dt = 1.0 / fs
        velocity = np.gradient(thorax_pc1, dt)

        logger.info("Using position derivative for velocity (fallback)")

    peak_vel_frame = int(np.argmax(velocity))
    peak_velocity = velocity[peak_vel_frame]

    logger.info(f"Peak velocity: {peak_velocity:.2f} m/s at frame {peak_vel_frame}")
    return velocity, peak_vel_frame, peak_velocity


def detect_strides(marker_data, peak_vel_frame, n_strides=N_STRIDES_TARGET):
    """
    Detect strides using right foot vertical position minima.

    Parameters
    ----------
    marker_data : np.ndarray
        Shape (n_frames, 23, 3).
    peak_vel_frame : int
        Frame of peak velocity.
    n_strides : int
        Target number of strides to extract.

    Returns
    -------
    stride_boundaries : list[tuple]
        List of (start_frame, end_frame) for each stride.
    contact_frames : np.ndarray
        All detected foot contact frames.
    """
    logger = logging.getLogger('sprint_pipeline')

    # Right foot vertical position (PC3 = z after PCA alignment)
    r_foot_z = marker_data[:, IDX_R_FOOT, 2]

    # Find minima (foot contacts) by inverting and finding peaks
    inverted_z = -r_foot_z
    peaks, properties = find_peaks(
        inverted_z,
        distance=STRIDE_MIN_DISTANCE,
        prominence=STRIDE_PROMINENCE_M
    )

    logger.info(f"Found {len(peaks)} foot contacts at frames: {peaks.tolist()}")

    if len(peaks) < 2:
        logger.error("Not enough foot contacts detected for stride segmentation")
        return [], peaks

    # Find the contact closest to peak velocity
    closest_idx = np.argmin(np.abs(peaks - peak_vel_frame))

    # Select n_strides+1 boundaries centered around peak velocity
    # Need 6 boundaries for 5 strides
    n_boundaries = n_strides + 1
    half_before = n_boundaries // 2
    half_after = n_boundaries - half_before

    start_idx = closest_idx - half_before
    end_idx = closest_idx + half_after

    # Clamp to available range
    if start_idx < 0:
        offset = -start_idx
        start_idx += offset
        end_idx += offset
    if end_idx > len(peaks):
        offset = end_idx - len(peaks)
        end_idx -= offset
        start_idx -= offset
    start_idx = max(0, start_idx)
    end_idx = min(len(peaks), end_idx)

    selected_contacts = peaks[start_idx:end_idx]
    actual_strides = len(selected_contacts) - 1

    if actual_strides < 1:
        logger.error("Could not extract any complete strides")
        return [], peaks

    if actual_strides < n_strides:
        logger.warning(f"Only found {actual_strides} strides (target: {n_strides})")

    # Build stride boundaries
    stride_boundaries = []
    for i in range(len(selected_contacts) - 1):
        stride_boundaries.append((selected_contacts[i], selected_contacts[i + 1]))

    for i, (s, e) in enumerate(stride_boundaries):
        logger.info(f"  Stride {i + 1}: frames {s}-{e} ({e - s} frames)")

    return stride_boundaries, peaks


def time_normalize_stride(stride_data, target_frames=TN_POINTS):
    """
    Time-normalize a stride to target_frames using linear interpolation.

    This is a Python re-implementation of the rubberband approach:
    ratio-based linear interpolation between adjacent frames.

    Parameters
    ----------
    stride_data : np.ndarray
        Shape (n_frames, 23, 3).
    target_frames : int
        Number of output frames (default: 101).

    Returns
    -------
    np.ndarray
        Shape (target_frames, 23, 3).
    """
    n_frames, n_markers, n_dims = stride_data.shape
    normalized = np.zeros((target_frames, n_markers, n_dims))
    ratio = (n_frames - 1) / (target_frames - 1)

    for i in range(target_frames):
        s = ratio * i
        if i == target_frames - 1:
            j = int(np.floor(s - 0.5))
        else:
            j = int(np.round(s - 0.5))
        j = max(0, min(j, n_frames - 2))
        f = s - j
        normalized[i, :, :] = (1 - f) * stride_data[j, :, :] + f * stride_data[j + 1, :, :]

    return normalized


def remove_translation(stride_data):
    """
    Remove forward translation by subtracting T12 position from all markers.

    Parameters
    ----------
    stride_data : np.ndarray
        Shape (n_frames, 23, 3).

    Returns
    -------
    np.ndarray
        Translation-removed data, same shape.
    """
    t12_pos = stride_data[:, IDX_T12, :]  # (frames, 3)
    debiased = stride_data - t12_pos[:, np.newaxis, :]
    return debiased


def create_stride_vector(mean_stride, height=None):
    """
    Scale by height and reshape into a 1D feature vector.

    Parameters
    ----------
    mean_stride : np.ndarray
        Shape (101, 23, 3).
    height : float or None
        Participant height in meters. If None, uses pelvis-to-head distance.

    Returns
    -------
    vector : np.ndarray
        Shape (6969,).
    height_used : float
        The height value used for scaling.
    """
    logger = logging.getLogger('sprint_pipeline')

    if height is None:
        # Estimate height from pelvis-to-head distance
        pelvis_pos = mean_stride[:, IDX_PELVIS, :]
        head_pos = mean_stride[:, IDX_HEAD, :]
        height = float(np.mean(np.sqrt(np.sum((head_pos - pelvis_pos) ** 2, axis=1))))
        logger.info(f"Height estimated from pelvis-to-head distance: {height:.3f} m")

    # Scale by height
    scaled = mean_stride / height

    # Vectorize: for each axis, stack all segments × timepoints
    # Order: [seg0_x(101), seg1_x(101), ..., seg22_x(101),
    #          seg0_y(101), seg1_y(101), ..., seg22_y(101),
    #          seg0_z(101), seg1_z(101), ..., seg22_z(101)]
    x_all = scaled[:, :, 0].T.flatten()  # (23*101,)
    y_all = scaled[:, :, 1].T.flatten()  # (23*101,)
    z_all = scaled[:, :, 2].T.flatten()  # (23*101,)

    vector = np.concatenate([x_all, y_all, z_all])
    logger.debug(f"Feature vector length: {len(vector)}")

    return vector, height


def process_trial(trial_info, fs=SAMPLING_RATE):
    """
    Run the full preprocessing pipeline for a single trial.

    Parameters
    ----------
    trial_info : dict
        Keys: participant_id, filepath, filetype.
    fs : int
        Sampling rate.

    Returns
    -------
    dict or None
        Result dictionary with keys: participant_id, vector, max_velocity,
        height, n_strides, filepath. Returns None on failure.
    """
    logger = logging.getLogger('sprint_pipeline')
    pid = trial_info['participant_id']
    filepath = trial_info['filepath']
    filetype = trial_info['filetype']

    logger.info(f"\n{'=' * 60}")
    logger.info(f"PROCESSING: {pid} ({filetype})")
    logger.info(f"File: {filepath}")
    logger.info(f"{'=' * 60}")

    try:
        # Step 1: Load data
        marker_data, meta = load_marker_data(filepath, filetype)
        raw_velocity_data = meta.get('velocity_data', None)
        logger.info(f"Step 1 - Loaded: {marker_data.shape[0]} frames, "
                    f"{marker_data.shape[1]} segments"
                    f"{', with Segment Velocity' if raw_velocity_data is not None else ''}")

        # Step 2: PCA alignment on FULL raw data FIRST
        # This must happen before sprint detection because IMU drift
        # causes the sprint direction to be split across raw axes.
        # After PCA: PC1 = sprint direction, PC2 = mediolateral, PC3 = vertical
        marker_data, pca, pca_rotation = perform_pca_alignment(marker_data)
        logger.info(f"Step 2 - PCA aligned (variance: "
                    f"{pca.explained_variance_ratio_[0]:.3f}, "
                    f"{pca.explained_variance_ratio_[1]:.3f}, "
                    f"{pca.explained_variance_ratio_[2]:.3f})")

        # Step 3: Detect sprint start using PC1 velocity (now aligned)
        start_frame = detect_sprint_start(marker_data, fs)
        marker_data = marker_data[start_frame:, :, :]
        # Crop velocity data to match
        if raw_velocity_data is not None:
            raw_velocity_data = raw_velocity_data[start_frame:, :, :]
        logger.info(f"Step 3 - Cropped to sprint: {marker_data.shape[0]} frames remaining")

        # Step 4: Shift origin (in PCA-aligned space)
        marker_data = shift_origin(marker_data)
        logger.info("Step 4 - Origin shifted to posterior foot")

        # Step 5: Trim to 60m
        n_before_trim = marker_data.shape[0]
        marker_data = trim_to_60m(marker_data)
        n_after_trim = marker_data.shape[0]
        # Crop velocity data to match trim
        if raw_velocity_data is not None:
            raw_velocity_data = raw_velocity_data[:n_after_trim, :, :]
        logger.info(f"Step 5 - Trimmed to 60m: {marker_data.shape[0]} frames")

        # Step 6: Compute velocity using XSens Segment Velocity if available
        velocity, peak_vel_frame, peak_velocity = compute_velocity(
            marker_data, fs,
            velocity_data=raw_velocity_data,
            pca_rotation=pca_rotation
        )
        logger.info(f"Step 6 - Peak velocity: {peak_velocity:.2f} m/s at frame {peak_vel_frame}")

        # Step 7: Detect strides
        stride_boundaries, contact_frames = detect_strides(marker_data, peak_vel_frame)
        n_strides = len(stride_boundaries)
        logger.info(f"Step 7 - Detected {n_strides} strides")

        if n_strides == 0:
            logger.error(f"No strides detected for {pid}, skipping")
            return None

        # Step 8: Time-normalize each stride
        normalized_strides = []
        for i, (s, e) in enumerate(stride_boundaries):
            stride = marker_data[s:e + 1, :, :]
            norm_stride = time_normalize_stride(stride, TN_POINTS)
            normalized_strides.append(norm_stride)
            logger.info(f"Step 8 - Stride {i + 1}: {e - s + 1} -> {TN_POINTS} frames")

        # Step 9: Remove translation (T12) from each stride
        debiased_strides = []
        for stride in normalized_strides:
            debiased_strides.append(remove_translation(stride))
        logger.info("Step 9 - Translation removed (T12 subtracted)")

        # Step 10: Ensemble average
        stacked = np.stack(debiased_strides, axis=0)  # (n_strides, 101, 23, 3)
        mean_stride = np.mean(stacked, axis=0)         # (101, 23, 3)
        logger.info(f"Step 10 - Ensemble averaged {n_strides} strides")

        # Step 11-12: Scale and vectorize
        vector, height_used = create_stride_vector(mean_stride, height=None)
        logger.info(f"Step 11-12 - Height: {height_used:.3f}m, Vector: {len(vector)} features")

        return {
            'participant_id': pid,
            'vector': vector,
            'max_velocity': peak_velocity,
            'height': height_used,
            'n_strides': n_strides,
            'filepath': filepath
        }

    except Exception as e:
        logger.error(f"FAILED processing {pid}: {e}")
        logger.debug(traceback.format_exc())
        return None


def build_dataset(root_dir):
    """
    Build the complete dataset by processing all valid trial files.

    Parameters
    ----------
    root_dir : str or Path
        Root directory containing trial data.
    """
    root_dir = Path(root_dir)
    output_dir = root_dir.parent / 'Sprints Code' / 'processed_data'
    logger = setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("SPRINT PREPROCESSING PIPELINE")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Root directory: {root_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)

    # Also scan the Chris V. Files directory for c3d/xlsx
    parent_dir = root_dir.parent
    search_dirs = [root_dir]
    chris_v_data = parent_dir / 'Chris V. Files' / '01 - Data'
    if chris_v_data.exists():
        search_dirs.append(chris_v_data)
        logger.info(f"Also scanning: {chris_v_data}")

    # Find all trial files across search directories
    all_trials = []
    seen_pids = set()
    for search_dir in search_dirs:
        trials = find_trial_files(search_dir)
        for t in trials:
            if t['participant_id'] not in seen_pids:
                all_trials.append(t)
                seen_pids.add(t['participant_id'])

    if len(all_trials) == 0:
        logger.error("No trial files found! Ensure .c3d or .xlsx files are in the directory.")
        return

    logger.info(f"\nTotal unique participants: {len(all_trials)}")

    # Process each trial
    results = []
    for i, trial_info in enumerate(all_trials):
        logger.info(f"\n>>> Processing {i + 1}/{len(all_trials)}: {trial_info['participant_id']}")
        result = process_trial(trial_info)
        if result is not None:
            results.append(result)
            logger.info(f">>> SUCCESS: {result['participant_id']}")
        else:
            logger.warning(f">>> FAILED: {trial_info['participant_id']}")

    # Build output matrix
    if len(results) == 0:
        logger.error("No participants were successfully processed!")
        return

    n_features = len(results[0]['vector'])
    dataset_matrix = np.zeros((len(results), n_features))
    for i, result in enumerate(results):
        dataset_matrix[i, :] = result['vector']

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Participant vectors
    npy_path = output_dir / 'participant_vectors.npy'
    np.save(npy_path, dataset_matrix)
    logger.info(f"\nSaved participant_vectors.npy: shape={dataset_matrix.shape}")

    # 2. Metadata CSV
    metadata_rows = []
    for r in results:
        metadata_rows.append({
            'participant_id': r['participant_id'],
            'max_velocity_ms': r['max_velocity'],
            'height_m': r['height'],
            'n_strides': r['n_strides'],
            'filepath': r['filepath']
        })

    metadata_df = pd.DataFrame(metadata_rows)
    csv_path = output_dir / 'metadata.csv'
    metadata_df.to_csv(csv_path, index=False)
    logger.info(f"Saved metadata.csv: {len(metadata_df)} participants")

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("PIPELINE COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Participants processed: {len(results)}/{len(all_trials)}")
    logger.info(f"Feature vector size: {n_features}")
    logger.info(f"Dataset shape: {dataset_matrix.shape}")
    logger.info(f"\nOutput files:")
    logger.info(f"  {npy_path}")
    logger.info(f"  {csv_path}")
    logger.info(f"  {output_dir / 'logs.txt'}")

    logger.info(f"\nPer-participant summary:")
    for r in results:
        logger.info(f"  {r['participant_id']}: "
                    f"v_max={r['max_velocity']:.2f} m/s, "
                    f"height={r['height']:.3f} m, "
                    f"strides={r['n_strides']}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    """Entry point for the preprocessing pipeline."""
    ROOT = "/Users/shunchen/Desktop/60m Project Folder/All MVN Files"
    build_dataset(ROOT)


if __name__ == "__main__":
    main()
