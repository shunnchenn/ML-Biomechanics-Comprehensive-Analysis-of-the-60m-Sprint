"""Bone connectivity for the 64-marker set, and 2-D skeleton drawing.

Polylines are transcribed from Chris Vellucci's MATLAB `xsens_scr_video.m` and
were 1-based there; they are converted once, here, and shared by every renderer.
"""
from __future__ import annotations

_AXIAL = [
    [2, 3, 5, 4, 2, 6, 7, 3, 7, 8, 6, 8, 9, 10, 11, 14, 16, 21, 12, 15, 11, 15,
     21, 15, 22, 15, 12, 22, 16],
    [22, 11], [21, 11],
    [17, 18, 19, 17, 20, 18, 20, 19, 20, 16],
]
_RIGHT = [
    [21, 23, 24, 29, 24, 21], [21, 29, 27, 29, 28, 27], [24, 27, 24, 28, 27],
    [23, 27, 23, 28, 27], [27, 33, 28, 33, 34, 33, 35, 33, 34, 27],
    [39, 40, 39, 41, 39, 42, 41, 40, 42, 47, 40, 47, 41, 47],
    [47, 48, 47, 49], [40, 48, 40, 49], [41, 48, 41, 49],
    [48, 49, 53, 48, 53, 54, 53, 55, 53, 56, 57, 56, 58, 53],
]
_LEFT = [
    [22, 25, 26, 25, 32, 22], [22, 26],
    [30, 31, 25, 30, 25, 31, 25, 26, 30, 26, 31, 26],
    [32, 30, 31, 30, 32, 31, 32],
    [30, 36, 30, 37, 30, 38, 36, 37, 31, 36, 31, 37, 31, 38, 31],
    [43, 44, 43, 45, 43, 46, 45, 44, 46], [50, 46, 50, 45, 50, 44, 50],
    [50, 51, 50, 52, 50], [44, 51, 44, 52, 44], [45, 51, 45, 52, 45],
    [51, 52, 51, 59, 52, 59, 60, 59, 60, 59, 62, 59, 61, 62, 63, 62, 64],
]

def _z(polys):
    """MATLAB 1-based marker numbers to Python 0-based."""
    return [[i - 1 for i in poly] for poly in polys]


AXIAL_IDX, RIGHT_IDX, LEFT_IDX = _z(_AXIAL), _z(_RIGHT), _z(_LEFT)
ALL_IDX = AXIAL_IDX + RIGHT_IDX + LEFT_IDX

COL_AXIAL, COL_RIGHT, COL_LEFT = "k", "#d62728", "#2ca02c"


def draw(ax, pts, plane="xz", color=None, lw=1.2, alpha=1.0):
    """Draw one 64-marker pose. plane 'xz' is sagittal, 'yz' frontal.

    Markers that are exactly at the origin were never populated (a dropped
    marker, or a reduced marker set); they are blanked so no bone is drawn back
    to (0, 0, 0).
    """
    import numpy as np

    pts = np.where((pts == 0).all(axis=-1, keepdims=True), np.nan, pts)
    h, v = (pts[:, 0], pts[:, 2]) if plane == "xz" else (pts[:, 1], pts[:, 2])
    for polys, col in ((AXIAL_IDX, COL_AXIAL), (RIGHT_IDX, COL_RIGHT), (LEFT_IDX, COL_LEFT)):
        for p in polys:
            ax.plot(h[p], v[p], color=color or col, lw=lw, alpha=alpha,
                    solid_capstyle="round")
