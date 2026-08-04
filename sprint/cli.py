"""python -m sprint build | analyse [--phase accel|topspeed]"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import config as C
from . import features, figures, frame, io, model

PHASES = ("accel", "topspeed")
TARGET = {"topspeed": "peak_vel", "accel": "t_0_10"}
# Faster acceleration is a *lower* 0-10 m split, so the sign is flipped to keep
# "positive coefficient = faster" true in both phases.
SIGN = {"topspeed": 1.0, "accel": -1.0}


def build(args):
    """C3D -> phase-normalised curves, per-step scalars and a QA report."""
    paths = sorted(C.C3D_DIR.glob("*.c3d"))
    if not paths:
        sys.exit(f"no .c3d files in {C.C3D_DIR} (set SPRINT_C3D_DIR)")
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)

    out = {p: {"rows": [], "curves": [], "tensor": []} for p in PHASES}
    qa = []
    for path in paths:
        pid = path.stem.split("-")[0].strip()
        if pid in C.EXCLUDED_PIDS:
            continue
        markers, labels, fs = io.load_c3d(path)
        idx = io.role_index(labels)
        for phase in PHASES:
            try:
                sc, curves, rows, rep = features.build(markers, idx, fs, phase)
            except Exception as exc:
                qa.append(dict(participant_id=pid, phase=phase, error=str(exc)))
                continue
            fwd = frame.heading(markers, idx, rows.td.iloc[0], rows.next_td.iloc[-1] + 1) \
                if phase == "accel" else None
            tensor = features.step_tensor(markers, idx, rows, fwd, rep["stature"])
            out[phase]["rows"].append(dict(sc, participant_id=pid,
                                           sex=C.PARTICIPANT_SEX.get(pid, "?"),
                                           peak_vel=rep["peak_vel"]))
            out[phase]["curves"].append(curves)
            out[phase]["tensor"].append(np.nanmean(tensor, axis=0))
            qa.append(dict(participant_id=pid, phase=phase, error="", **rep))

    for phase in PHASES:
        d = out[phase]
        if not d["rows"]:
            continue
        pd.DataFrame(d["rows"]).to_csv(C.DATA_DIR / f"features_{phase}.csv", index=False)
        np.save(C.DATA_DIR / f"curves_{phase}.npy", np.stack(d["curves"]))
        np.save(C.DATA_DIR / f"poses_{phase}.npy", np.stack(d["tensor"]))
    pd.DataFrame(qa).to_csv(C.DATA_DIR / "qa_report.csv", index=False)
    _print_qa(pd.DataFrame(qa))


def _print_qa(qa):
    """Surface the physiological checks rather than burying them in a file."""
    ok = qa[qa.error == ""]
    if ok.empty:
        print("no trials processed")
        return
    print(f"trials processed: {len(ok)}  failed: {int((qa.error != '').sum())}")
    for col in ("gct_in_range", "duty_in_range", "v_matches_sl_x_sf"):
        bad = ok[~ok[col].astype(bool)]
        print(f"  {col}: {len(ok) - len(bad)}/{len(ok)} pass"
              + (f"  failing: {sorted(bad.participant_id.unique())}" if len(bad) else ""))
    print(f"  mean GCT {ok.mean_gct_s.mean():.3f}s  mean duty {ok.mean_duty.mean():.3f}")


def _load(phase):
    df = pd.read_csv(C.DATA_DIR / f"features_{phase}.csv")
    curves = np.load(C.DATA_DIR / f"curves_{phase}.npy")
    poses = np.load(C.DATA_DIR / f"poses_{phase}.npy")
    if TARGET[phase] not in df.columns:
        splits = pd.read_csv(C.DATA_DIR / "split_times.csv")
        df = df.merge(splits[["participant_id", TARGET[phase]]], on="participant_id")
    return df, curves, poses


def analyse(args):
    """Fit the interpretable models and render the figures for one phase."""
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    for phase in ([args.phase] if args.phase else PHASES):
        df, curves, poses = _load(phase)
        y = SIGN[phase] * df.pop(TARGET[phase]).values
        names = [c for c in df.columns
                 if df[c].dtype.kind == "f" and c not in ("peak_vel",)]
        X = np.nan_to_num(df[names].values.astype(float))
        fast = y >= np.median(y)

        en = model.elasticnet(X, y, names)
        ci = model.bootstrap_ci(X, y, names, en["model"], n_boot=args.n_boot)
        lg = model.logistic_tertile(X, y, names)
        shap_vals = model.shap_linear(en["model"], X)

        label = {"accel": "Acceleration, first 4 steps (faster = lower 0-10 m split)",
                 "topspeed": "Top speed (faster = higher peak velocity)"}[phase]
        figures.phase_ribbon(curves, fast, label,
                             C.FIG_DIR / f"optimal_kinematics_{phase}.png", args.n_perm)
        figures.coefficient_plot(en, ci, lg, label,
                                 C.FIG_DIR / f"coefficients_{phase}.png")
        figures.kinogram(poses, fast, label, C.FIG_DIR / f"kinogram_{phase}.png")

        shap_mean = pd.DataFrame({"feature": names,
                                  "mean_abs_shap": np.abs(shap_vals).mean(axis=0)})
        out = en["coef"].merge(ci, on="feature").merge(lg["coef"], on="feature") \
                        .merge(shap_mean, on="feature")
        out.to_csv(C.DATA_DIR / f"interpretation_{phase}.csv", index=False)
        print(f"{phase}: ElasticNet LOO R2 = {en['r2_loo']:.3f}, RMSE = {en['rmse']:.3f}; "
              f"logistic LOO AUC = {lg['auc_loo']:.3f} "
              f"(n={lg['n_fast']} fast / {lg['n_slow']} slow)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="sprint")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(fn=build)
    a = sub.add_parser("analyse")
    a.add_argument("--phase", choices=PHASES)
    a.add_argument("--n-boot", type=int, default=2000)
    a.add_argument("--n-perm", type=int, default=5000)
    a.set_defaults(fn=analyse)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
