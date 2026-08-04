"""C3D loading and marker-role resolution."""
from __future__ import annotations

import numpy as np

from . import config as C


def load_c3d(path):
    """Read one trial. Returns (markers (n_frames, n_markers, 3) in metres, labels, fs)."""
    import ezc3d

    c = ezc3d.c3d(str(path))
    raw = c["data"]["points"][:3].transpose(2, 1, 0).astype(float)
    labels = list(c["parameters"]["POINT"]["LABELS"]["value"])
    fs = int(c["header"]["points"]["frame_rate"])

    # Gap-fill: interpolate each marker/axis across frames where tracking dropped.
    t = np.arange(raw.shape[0])
    for m in range(raw.shape[1]):
        for a in range(3):
            col = raw[:, m, a]
            bad = np.isnan(col)
            if bad.any() and not bad.all():
                col[bad] = np.interp(t[bad], t[~bad], col[~bad])

    if np.nanmax(np.abs(raw)) > 100:      # some files store millimetres
        raw /= 1000.0
    return raw, labels, fs


def role_index(labels=None):
    """Map each marker role to the column indices that define it.

    Resolves by c3d label when labels are given, else falls back to the verified
    numeric indices in config. Raises if a role cannot be resolved, so a renamed
    marker set fails loudly instead of silently reading the wrong column.
    """
    if labels is None:
        return {k: np.array(v) for k, v in C.MARKER_IDX.items()}

    pos = {lbl: i for i, lbl in enumerate(labels)}
    out, missing = {}, []
    for role, spec in C.LABELS.items():
        names = (spec,) if isinstance(spec, str) else spec
        idx = [pos[n] for n in names if n in pos]
        if len(idx) != len(names):
            missing.append(role)
        else:
            out[role] = np.array(idx)
    if missing:
        raise KeyError(f"unresolved marker roles {missing}; labels present: {len(pos)}")
    return out


def points(markers, idx, role):
    """Position of one role over time: (n_frames, 3), averaging its markers."""
    return markers[:, idx[role], :].mean(axis=1)
