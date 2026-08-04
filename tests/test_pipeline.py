"""Pins the alignment, event and model code without touching participant data.

The synthetic trial has known contact and flight durations, a known stature and a
known heading, and is then rotated by an arbitrary yaw and given lateral drift.
Anything that recovers the truth has to do so in spite of both.
"""
from __future__ import annotations

import numpy as np
import pytest

from sprint import config as C
from sprint import events, features, frame, io, model
from tests.synthetic import make_trial

CASES = [
    dict(),
    dict(contact=6, flight=8, v=10.0, yaw_deg=-112.0, seed=3),
    dict(contact=11, flight=5, v=6.0, yaw_deg=200.0, drift=1.5, seed=7),
    dict(contact=8, flight=6, v=8.2, yaw_deg=0.0, drift=0.0, noise=0.004, seed=11),
    dict(contact=5, flight=9, v=10.5, yaw_deg=88.0, drift=2.0, noise=0.006, seed=13),
]


@pytest.fixture(params=CASES, ids=lambda k: str(k) or "default")
def trial(request):
    markers, truth = make_trial(**request.param)
    return markers, io.role_index(), truth


def test_stature_recovered(trial):
    markers, idx, truth = trial
    assert frame.stature(markers, idx) == pytest.approx(truth["stature"], abs=0.05)


def test_local_frame_is_yaw_invariant():
    """The same motion recorded at two lab orientations must align identically."""
    a, ta = make_trial(yaw_deg=0.0, drift=0.0, noise=0.0, seed=5)
    b, _ = make_trial(yaw_deg=143.0, drift=0.0, noise=0.0, seed=5)
    idx = io.role_index()
    la, lb = (frame.to_local(m, frame.heading(m, idx), m[0, 0], 1.0) for m in (a, b))
    assert np.abs(la - lb).max() < 1e-6


def test_events_exact(trial):
    """Touchdown and toe-off to the frame, for every contact/flight ratio."""
    markers, idx, truth = trial
    ev = events.detect(markers, idx, truth["fs"])
    for key, col in (("td", 0), ("to", 1)):
        found = sorted(set(ev["R"][col]) | set(ev["L"][col]))
        assert found, f"no {key} detected"
        assert max(min(abs(np.array(truth[key]) - f)) for f in found) <= 1


def test_step_timings_match_truth(trial):
    markers, idx, truth = trial
    df = events.step_table(markers, idx, truth["fs"])
    v = df[df.valid]
    assert len(v) >= 5
    assert v.t_contact.mean() == pytest.approx(truth["contact_s"], abs=1 / truth["fs"])
    assert v.t_flight.mean() == pytest.approx(truth["flight_s"], abs=1 / truth["fs"])
    assert v.step_length.mean() == pytest.approx(truth["step_length"], rel=0.02)


def test_qa_catches_velocity_inconsistency(trial):
    """v = step length x step frequency must close — the check the old GCT failed."""
    markers, idx, truth = trial
    rep = events.qa(events.step_table(markers, idx, truth["fs"]), truth["v"])
    assert rep["v_matches_sl_x_sf"]
    assert rep["sl_x_sf_ms"] == pytest.approx(truth["v"], rel=C.V_SLSF_TOL)


def test_phase_base_shape_and_short_phase_is_nan(trial):
    markers, idx, truth = trial
    df = events.step_table(markers, idx, truth["fs"])
    row = next(df[df.valid].itertuples())
    P = events.phase_normalise(markers, row)
    assert P.shape == (C.N_PHASE, markers.shape[1], 3)
    assert not np.isnan(P).all()

    short = row._replace(to=row.td + 1, next_td=row.td + 2)
    assert np.isnan(events.phase_normalise(markers, short)).all()


def test_features_build_both_phases(trial):
    markers, idx, truth = trial
    for phase in ("accel", "topspeed"):
        sc, curves, rows, rep = features.build(markers, idx, truth["fs"], phase)
        assert curves.shape == (C.N_PHASE, len(features.CURVES))
        assert np.isfinite(sc.values).all()
        assert rep["peak_vel"] == pytest.approx(truth["v"], rel=0.05)


def test_neutral_pose_angles_are_zero():
    """A standing pose must read zero at every joint, given the sign convention."""
    idx = io.role_index()
    P = np.zeros((1, C.N_MARKERS, 3))
    for role, z in (("pelvis", 1.0), ("R_shoulder", 1.4), ("L_shoulder", 1.4),
                    ("R_hip", 1.0), ("R_knee", 0.5), ("R_ankle", 0.1), ("R_heel", 0.05)):
        P[0, C.MARKER_IDX[role], 2] = z
    P[0, C.MARKER_IDX["R_toe"], :] = [0.2, 0.0, 0.05]
    a = features.angles(P, idx, "R")[0]
    for name in ("trunk", "thigh", "shank", "hip", "knee", "ankle"):
        assert a[features.CURVES.index(name)] == pytest.approx(0.0, abs=1e-6)


def test_spm_finds_injected_cluster_and_rejects_noise():
    rng = np.random.default_rng(0)
    curves = rng.normal(size=(30, C.N_PHASE))
    group = np.arange(30) < 15
    curves[group, 10:20] += 1.5
    _, clusters = model.spm(curves, group, n_perm=500)
    sig = [(a, b) for a, b, _, p in clusters if p < 0.05]
    assert len(sig) == 1
    assert sig[0][0] >= 9 and sig[0][1] <= 20


def test_elasticnet_and_logistic_run_on_real_committed_features():
    """Model layer against the real data that ships with the repo."""
    import pandas as pd

    path = C.DATA_DIR / "sprint_biomechanics_metrics.csv"
    if not path.exists():
        pytest.skip("committed metrics not present")
    bio = pd.read_csv(path)
    names = [c for c in bio.columns
             if bio[c].dtype.kind == "f" and c not in ("max_velocity_ms", "height_m")]
    X = np.nan_to_num(bio[names].values.astype(float))
    y = bio["max_velocity_ms"].values

    en = model.elasticnet(X, y, names)
    assert np.isfinite(en["r2_loo"]) and len(en["coef"]) == len(names)
    assert model.shap_linear(en["model"], X).shape == X.shape
    assert 0.0 <= model.logistic_tertile(X, y, names)["auc_loo"] <= 1.0
    # The interval must actually bracket the estimate it describes. Re-tuning the
    # penalty per resample breaks this, which is why bootstrap_ci takes the fit.
    ci = model.bootstrap_ci(X, y, names, en["model"], n_boot=200).set_index("feature")
    beta = en["coef"].set_index("feature").beta.reindex(ci.index)
    assert ((ci.lo - 1e-9 <= beta) & (beta <= ci.hi + 1e-9)).all()
