"""
_sprint_model_comparison.py
────────────────────────────────────────────────────────────────────────────
Standalone 24-model LOO-CV sprint performance comparison.

Loads pre-computed top-speed stride data from Outputs/Data/, runs fPCA,
builds four feature sets (fPC scores, biomechanics scalars, combined, raw),
evaluates all model × feature combinations with Leave-One-Out CV, and
produces two publication-quality figures:

  Outputs/Figures/ml_bio_vs_fpc_panel.png  — Bio vs fPC table + overfit curve
  Outputs/Figures/ml_model_ranking.png     — all models ranked by LOO-CV R²

Usage
-----
  python _sprint_model_comparison.py          # run from repo root
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (BayesianRidge, ElasticNetCV, HuberRegressor,
                                   LassoCV, LinearRegression, RidgeCV)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "Outputs" / "Data"
FIG_DIR    = SCRIPT_DIR / "Outputs" / "Figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load data ───────────────────────────────────────────────────────────
print("Loading data …")
dataset   = np.load(DATA_DIR / "participant_vectors.npy")          # (n, 19392)
meta_df   = pd.read_csv(DATA_DIR / "metadata.csv")
bio_df    = pd.read_csv(DATA_DIR / "sprint_biomechanics_metrics.csv")
pid_order = meta_df["participant_id"].tolist()
y         = meta_df["max_velocity_ms"].values.astype(float)
n_subj    = len(y)
print(f"  n={n_subj} subjects  |  dataset {dataset.shape}")

# ── 2. Top-speed fPCA ──────────────────────────────────────────────────────
print("Running fPCA …")
mu     = dataset.mean(axis=0)
fpca   = PCA(n_components=min(30, n_subj - 1, dataset.shape[1]),
             random_state=RANDOM_SEED)
scores = fpca.fit_transform(dataset - mu)          # (n, n_pc)
print(f"  fPC scores: {scores.shape}")

# ── 3. Stepwise PC selection ───────────────────────────────────────────────
def stepwise_pcs(scores_mat: np.ndarray, y_vec: np.ndarray,
                 max_pcs: int = 3, p_thresh: float = 0.10) -> list[int]:
    selected: list[int] = []
    remaining = list(range(scores_mat.shape[1]))
    for _ in range(max_pcs):
        best_p, best_pc = 1.0, None
        for pc in remaining:
            res = sm.OLS(y_vec,
                         sm.add_constant(scores_mat[:, selected + [pc]])).fit()
            if res.pvalues[-1] < best_p:
                best_p, best_pc = res.pvalues[-1], pc
        if best_pc is not None and best_p < p_thresh:
            selected.append(best_pc)
            remaining.remove(best_pc)
        else:
            break
    if not selected:
        selected = [int(np.argmax(
            [abs(pearsonr(scores_mat[:, i], y_vec)[0])
             for i in range(scores_mat.shape[1])]))]
    return selected

RETAINED_PCS = stepwise_pcs(scores, y)
print(f"  Stepwise retained PCs (0-indexed): {RETAINED_PCS}")

# ── 4. Build feature sets ──────────────────────────────────────────────────
# A — all fPC scores (29 PCs)
X_A = scores.copy()

# B — biomechanics scalars (all numeric, excluding target & participant id)
_excl = {"participant_id", "max_velocity_ms"}
bio_numeric_cols = [c for c in bio_df.columns
                    if bio_df[c].dtype.kind in "fi" and c not in _excl]
X_B = (bio_df.set_index("participant_id")
             .reindex(pid_order)[bio_numeric_cols]
             .values.astype(float))
X_B = np.nan_to_num(X_B, nan=0.0, posinf=0.0, neginf=0.0)

# C — combined (fPC + bio)
X_C = np.hstack([X_A, X_B])

print(f"  X_A (fPC):      {X_A.shape}")
print(f"  X_B (Bio):      {X_B.shape}  cols: {bio_numeric_cols}")
print(f"  X_C (Combined): {X_C.shape}")
print(f"  X_raw (Raw):    {dataset.shape}")

# ── 5. LOO-CV evaluation harness ───────────────────────────────────────────
loo = LeaveOneOut()

def eval_model(name: str, pipeline, X: np.ndarray, y_true: np.ndarray,
               feat_label: str) -> dict:
    t0     = time.time()
    y_pred = cross_val_predict(pipeline, X, y_true, cv=loo)
    r2_loo = float(r2_score(y_true, y_pred))
    rmse   = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    pipeline.fit(X, y_true)
    r2_tr  = float(r2_score(y_true, pipeline.predict(X)))
    return {
        "Model":       name,
        "Features":    feat_label,
        "R2_LOO":      round(r2_loo,       4),
        "RMSE":        round(rmse,         4),
        "R2_train":    round(r2_tr,        4),
        "Overfit_Gap": round(r2_tr - r2_loo, 4),
        "n_features":  X.shape[1],
        "time_s":      round(time.time() - t0, 1),
        "_y_pred":     y_pred,
    }

# ── 6. Run all models ──────────────────────────────────────────────────────
results: list[dict] = []
_step = [0]
_total = 26

def _log(msg: str) -> None:
    _step[0] += 1
    print(f"  {_step[0]:2d}/{_total}  {msg}", flush=True)

print("\nRunning LOO-CV for all models …")

# 1. Stepwise OLS [fPC]
_log(f"Stepwise OLS [fPC ({len(RETAINED_PCS)} PCs)]")
results.append(eval_model(
    "Stepwise OLS",
    Pipeline([("m", LinearRegression())]),
    scores[:, RETAINED_PCS], y, f"fPC ({len(RETAINED_PCS)} PCs)"))

# 2. OLS [fPC (all), Bio]
for X_dat, fl in [(X_A, "fPC (all)"), (X_B, "Bio")]:
    _log(f"OLS [{fl}]")
    results.append(eval_model(
        "OLS",
        Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]),
        X_dat, y, fl))

# 3. Ridge [fPC (all), Bio, Combined]
for X_dat, fl in [(X_A, "fPC (all)"), (X_B, "Bio"), (X_C, "Combined")]:
    _log(f"Ridge [{fl}]")
    results.append(eval_model(
        "Ridge",
        Pipeline([("sc", StandardScaler()),
                  ("m",  RidgeCV(alphas=np.logspace(-4, 4, 80)))]),
        X_dat, y, fl))

# 4. LASSO [fPC (all), Bio, Combined]
for X_dat, fl in [(X_A, "fPC (all)"), (X_B, "Bio"), (X_C, "Combined")]:
    _log(f"LASSO [{fl}]")
    results.append(eval_model(
        "LASSO",
        Pipeline([("sc", StandardScaler()),
                  ("m",  LassoCV(alphas=np.logspace(-5, 2, 80), cv=5,
                                 random_state=RANDOM_SEED, max_iter=10000))]),
        X_dat, y, fl))

# 5. Elastic Net [fPC (all), Bio]
for X_dat, fl in [(X_A, "fPC (all)"), (X_B, "Bio")]:
    _log(f"Elastic Net [{fl}]")
    results.append(eval_model(
        "Elastic Net",
        Pipeline([("sc", StandardScaler()),
                  ("m",  ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95],
                                      alphas=np.logspace(-4, 1, 30), cv=5,
                                      random_state=RANDOM_SEED, max_iter=10000))]),
        X_dat, y, fl))

# 6. Bayesian Ridge [fPC (all), Bio]
for X_dat, fl in [(X_A, "fPC (all)"), (X_B, "Bio")]:
    _log(f"Bayesian Ridge [{fl}]")
    results.append(eval_model(
        "Bayesian Ridge",
        Pipeline([("sc", StandardScaler()), ("m", BayesianRidge())]),
        X_dat, y, fl))

# 7. Huber [fPC (all), Bio, Combined]
for X_dat, fl in [(X_A, "fPC (all)"), (X_B, "Bio"), (X_C, "Combined")]:
    _log(f"Huber [{fl}]")
    results.append(eval_model(
        "Huber",
        Pipeline([("sc", StandardScaler()),
                  ("m",  HuberRegressor(epsilon=1.35, max_iter=200))]),
        X_dat, y, fl))

# 8. Random Forest [fPC (all), Bio]
for X_dat, fl in [(X_A, "fPC (all)"), (X_B, "Bio")]:
    _log(f"Random Forest [{fl}]")
    results.append(eval_model(
        "Random Forest",
        Pipeline([("sc", StandardScaler()),
                  ("m",  RandomForestRegressor(n_estimators=500, max_depth=5,
                                               min_samples_leaf=2,
                                               random_state=RANDOM_SEED))]),
        X_dat, y, fl))

# 9. XGBoost [fPC (all), Bio]
for X_dat, fl in [(X_A, "fPC (all)"), (X_B, "Bio")]:
    _log(f"XGBoost [{fl}]")
    results.append(eval_model(
        "XGBoost",
        Pipeline([("sc", StandardScaler()),
                  ("m",  xgb.XGBRegressor(n_estimators=100, max_depth=3,
                                           learning_rate=0.1, reg_alpha=1.0,
                                           reg_lambda=1.0, verbosity=0,
                                           random_state=RANDOM_SEED))]),
        X_dat, y, fl))

# 10. SVR – grid-search C separately for each feature set
for X_dat, fl in [(X_A, "fPC (all)"), (X_B, "Bio")]:
    _log(f"SVR [{fl}]")
    best_r2, best_C = -999.0, 1.0
    for C in [0.1, 1.0, 10.0]:
        yp  = cross_val_predict(
            Pipeline([("sc", StandardScaler()), ("m", SVR(C=C))]),
            X_dat, y, cv=loo)
        r2v = r2_score(y, yp)
        if r2v > best_r2:
            best_r2, best_C = r2v, C
    results.append(eval_model(
        f"SVR (C={best_C:.0f})",
        Pipeline([("sc", StandardScaler()), ("m", SVR(C=best_C))]),
        X_dat, y, fl))

# 11. kNN – grid-search k separately for each feature set
for X_dat, fl in [(X_A, "fPC (all)"), (X_B, "Bio")]:
    _log(f"kNN [{fl}]")
    best_r2, best_k = -999.0, 5
    for k in [3, 5, 7, 9]:
        yp  = cross_val_predict(
            Pipeline([("sc", StandardScaler()),
                      ("m",  KNeighborsRegressor(n_neighbors=k))]),
            X_dat, y, cv=loo)
        r2v = r2_score(y, yp)
        if r2v > best_r2:
            best_r2, best_k = r2v, k
    results.append(eval_model(
        f"kNN (k={best_k})",
        Pipeline([("sc", StandardScaler()),
                  ("m",  KNeighborsRegressor(n_neighbors=best_k))]),
        X_dat, y, fl))

# 12. PCR – grid-search n_components on raw data
_log("PCR [Raw]")
best_r2, best_n = -999.0, 1
for n_comp in range(1, 16):
    yp  = cross_val_predict(
        Pipeline([("sc",  StandardScaler()),
                  ("pca", PCA(n_components=n_comp)),
                  ("m",   LinearRegression())]),
        dataset, y, cv=loo)
    r2v = r2_score(y, yp)
    if r2v > best_r2:
        best_r2, best_n = r2v, n_comp
results.append(eval_model(
    f"PCR (n={best_n})",
    Pipeline([("sc",  StandardScaler()),
              ("pca", PCA(n_components=best_n)),
              ("m",   LinearRegression())]),
    dataset, y, f"Raw ({dataset.shape[1]})"))

# 13. PLS – grid-search n_components on raw data
_log("PLS [Raw]")
best_r2, best_n = -999.0, 1
for n_comp in range(1, 11):
    yp  = cross_val_predict(
        Pipeline([("sc", StandardScaler()),
                  ("m",  PLSRegression(n_components=n_comp))]),
        dataset, y, cv=loo)
    r2v = r2_score(y, yp)
    if r2v > best_r2:
        best_r2, best_n = r2v, n_comp
results.append(eval_model(
    f"PLS (n={best_n})",
    Pipeline([("sc", StandardScaler()),
              ("m",  PLSRegression(n_components=best_n))]),
    dataset, y, f"Raw ({dataset.shape[1]})"))

print(f"\n  → {len(results)} model-feature combinations evaluated.")

# ── 7. Results DataFrame ───────────────────────────────────────────────────
res_df = (pd.DataFrame([{k: v for k, v in r.items() if k != "_y_pred"}
                         for r in results])
            .sort_values("R2_LOO", ascending=False)
            .reset_index(drop=True))

csv_path = DATA_DIR / "ml_model_comparison.csv"
res_df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")
print(res_df[["Model", "Features", "R2_LOO", "RMSE", "Overfit_Gap"]].to_string(index=False))

# ── 8. Overfitting scan (Train vs LOO-CV R² by #PCs) ──────────────────────
print("\nRunning overfitting scan …")
pc_cors = sorted(
    [(i, abs(pearsonr(scores[:, i], y)[0])) for i in range(scores.shape[1])],
    key=lambda x: x[1], reverse=True)
top_pcs_by_cor = [x[0] for x in pc_cors]

n_scan = min(10, len(top_pcs_by_cor))
train_r2s, loo_r2s = [], []
for k in range(1, n_scan + 1):
    X_k = scores[:, top_pcs_by_cor[:k]]
    lr  = LinearRegression()
    lr.fit(X_k, y)
    train_r2s.append(r2_score(y, lr.predict(X_k)))
    loo_r2s.append(r2_score(y, cross_val_predict(lr, X_k, y, cv=loo)))

# ── 9. Figure 1 – Bio vs fPC table  +  Overfitting diagnostic ─────────────
# Identify matched algorithm pairs (same algo, Bio vs fPC-all)
_MATCHED_ALGOS = ["Ridge", "LASSO", "Elastic Net", "Bayesian Ridge",
                  "Random Forest", "XGBoost"]

def _lookup(df: pd.DataFrame, model: str, feat: str) -> float:
    mask = (df["Model"] == model) & (df["Features"] == feat)
    if mask.any():
        return float(df.loc[mask, "R2_LOO"].iloc[0])
    return float("nan")

bio_r2s = [_lookup(res_df, m, "Bio")       for m in _MATCHED_ALGOS]
fpc_r2s = [_lookup(res_df, m, "fPC (all)") for m in _MATCHED_ALGOS]
bio_wins = sum(b > f for b, f in zip(bio_r2s, fpc_r2s)
               if not (np.isnan(b) or np.isnan(f)))
bio_wins_all = bio_wins == sum(
    1 for b, f in zip(bio_r2s, fpc_r2s)
    if not (np.isnan(b) or np.isnan(f)))

fig1 = plt.figure(figsize=(15, 5.5))
gs1  = gridspec.GridSpec(1, 2, figure=fig1, wspace=0.35)

# ── Left: table ──────────────────────────────────────────────────────────
ax_tbl = fig1.add_subplot(gs1[0])
ax_tbl.set_facecolor("#f4f4f4")
ax_tbl.axis("off")

col_labels  = ["Algorithm", "Bio R²", "fPC R²"]
col_widths  = [0.44, 0.28, 0.28]
n_rows      = len(_MATCHED_ALGOS)
row_height  = 0.11
header_y    = 0.93

# Header
header_bg = mpatches.FancyBboxPatch(
    (0, header_y - 0.04), 1.0, 0.10,
    boxstyle="round,pad=0.005", facecolor="#1a2d5a", edgecolor="none",
    transform=ax_tbl.transAxes, clip_on=False)
ax_tbl.add_patch(header_bg)
x0 = 0.0
for lbl, w in zip(col_labels, col_widths):
    ax_tbl.text(x0 + w / 2, header_y + 0.01, lbl,
                ha="center", va="center", fontsize=11,
                fontweight="bold", color="white",
                transform=ax_tbl.transAxes)
    x0 += w

def _val_color(v: float) -> str:
    if np.isnan(v):
        return "#cccccc"
    if v > 0.15:
        return "#2ca02c"    # strong green
    if v > 0.0:
        return "#f4a62a"    # amber
    return "#d62728"        # red

# Data rows
for row_i, (algo, b_r2, f_r2) in enumerate(
        zip(_MATCHED_ALGOS, bio_r2s, fpc_r2s)):
    y_row = header_y - 0.04 - (row_i + 1) * row_height
    bg_col = "#ffffff" if row_i % 2 == 0 else "#f0f3fa"
    row_bg = mpatches.FancyBboxPatch(
        (0, y_row - 0.025), 1.0, row_height,
        boxstyle="round,pad=0.002", facecolor=bg_col, edgecolor="none",
        transform=ax_tbl.transAxes, clip_on=False)
    ax_tbl.add_patch(row_bg)
    # algo name
    ax_tbl.text(col_widths[0] / 2, y_row + row_height / 2 - 0.03, algo,
                ha="center", va="center", fontsize=10.5,
                fontweight="bold", color="#1a2d5a",
                transform=ax_tbl.transAxes)
    # Bio R²
    b_str = f"{b_r2:.3f}" if not np.isnan(b_r2) else "—"
    ax_tbl.text(col_widths[0] + col_widths[1] / 2,
                y_row + row_height / 2 - 0.03,
                b_str, ha="center", va="center",
                fontsize=11, fontweight="bold",
                color=_val_color(b_r2),
                transform=ax_tbl.transAxes)
    # fPC R²
    f_str = f"{f_r2:.3f}" if not np.isnan(f_r2) else "—"
    ax_tbl.text(col_widths[0] + col_widths[1] + col_widths[2] / 2,
                y_row + row_height / 2 - 0.03,
                f_str, ha="center", va="center",
                fontsize=11, fontweight="bold",
                color=_val_color(f_r2),
                transform=ax_tbl.transAxes)

# Summary banner
banner_y = header_y - 0.04 - (n_rows + 1.5) * row_height
if bio_wins_all:
    banner_txt = "Bio wins EVERY matched algorithm pair ✓"
    banner_fc  = "#d4edda"
    banner_ec  = "#28a745"
    banner_tc  = "#155724"
else:
    banner_txt = f"Bio wins {bio_wins}/{len(_MATCHED_ALGOS)} matched pairs"
    banner_fc  = "#fff3cd"
    banner_ec  = "#ffc107"
    banner_tc  = "#856404"

banner_box = mpatches.FancyBboxPatch(
    (0.01, banner_y - 0.02), 0.98, row_height,
    boxstyle="round,pad=0.01", facecolor=banner_fc, edgecolor=banner_ec, lw=1.5,
    transform=ax_tbl.transAxes, clip_on=False)
ax_tbl.add_patch(banner_box)
ax_tbl.text(0.5, banner_y + row_height / 2 - 0.04, banner_txt,
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color=banner_tc, transform=ax_tbl.transAxes)

ax_tbl.set_title("Bio vs fPC: Matched Algorithm Comparison\n"
                 r"Target: Peak Velocity (m/s)",
                 fontsize=12, fontweight="bold", pad=10)

# ── Right: Overfitting diagnostic ────────────────────────────────────────
ax_ov = fig1.add_subplot(gs1[1])
ks = range(1, n_scan + 1)
ax_ov.plot(ks, train_r2s, "o-",  color="#1f77b4", lw=2,   label="Train R²")
ax_ov.plot(ks, loo_r2s,   "s--", color="#d62728", lw=2,   label="LOO-CV R²")
ax_ov.fill_between(ks, loo_r2s, train_r2s,
                   alpha=0.14, color="#d62728", label="Overfit gap")
ax_ov.axhline(0, color="grey", ls=":", lw=0.8)
ax_ov.axvline(len(RETAINED_PCS), color="#ff7f0e", ls="--", lw=1.5,
              label=f"Stepwise selection ({len(RETAINED_PCS)} PCs)")
ax_ov.set_xlabel("Number of PCs in model (ranked by |r| with peak velocity)",
                 fontsize=10)
ax_ov.set_ylabel("R²  [target: peak velocity, m/s]", fontsize=10)
ax_ov.set_title("Overfitting Diagnostic: Train vs LOO-CV R²\n"
                r"Target: Peak Velocity (m/s)",
                fontweight="bold", fontsize=12)
ax_ov.legend(fontsize=9)
ax_ov.set_xticks(list(ks))
ax_ov.grid(alpha=0.3)
ax_ov.set_ylim(min(loo_r2s) - 0.05, max(train_r2s) + 0.05)

fig1.suptitle(f"Sprint Velocity Prediction  ·  Target: Peak Velocity (m/s)  ·  n={n_subj} subjects",
              fontsize=13, fontweight="bold", y=1.02)
fig1.tight_layout()
out1 = FIG_DIR / "ml_bio_vs_fpc_panel.png"
fig1.savefig(str(out1), dpi=150, bbox_inches="tight")
print(f"\nSaved: {out1}")
plt.close(fig1)

# ── 10. Figure 2 – All models ranked by LOO-CV R² ─────────────────────────
# Colour map: Bio=blue, fPC(all)=orange, fPC(step)=goldenrod,
#             Combined=green, Raw=grey
def _bar_color(feat: str) -> str:
    if feat == "Bio":
        return "#1f77b4"
    if feat == "fPC (all)":
        return "#d62728"
    if "PCs" in feat:                   # "fPC (3 PCs)"
        return "#e8a820"
    if feat == "Combined":
        return "#2ca02c"
    return "#888888"                    # Raw

labels  = [f"{r['Model']}\n[{r['Features']}]"
           for _, r in res_df.iterrows()]
r2_vals = res_df["R2_LOO"].values
colors  = [_bar_color(r["Features"]) for _, r in res_df.iterrows()]

fig2, ax2 = plt.subplots(figsize=(12, max(7, len(labels) * 0.38)))
y_pos = np.arange(len(labels))

bars = ax2.barh(y_pos, r2_vals, color=colors, edgecolor="white",
                linewidth=0.6, alpha=0.88, height=0.72)

# Value labels on bars
for bar, val in zip(bars, r2_vals):
    x_label = val + 0.005 if val >= 0 else val - 0.005
    ha      = "left"      if val >= 0 else "right"
    ax2.text(x_label, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", ha=ha, fontsize=8.5, fontweight="bold")

# Best-in-class dashed lines
bio_best  = max((r["R2_LOO"] for _, r in res_df.iterrows()
                 if r["Features"] == "Bio"), default=np.nan)
fpc_best  = max((r["R2_LOO"] for _, r in res_df.iterrows()
                 if r["Features"] == "fPC (all)"), default=np.nan)
comb_best = max((r["R2_LOO"] for _, r in res_df.iterrows()
                 if r["Features"] == "Combined"), default=np.nan)

if not np.isnan(bio_best):
    ax2.axvline(bio_best, color="#1f77b4", lw=1.2, ls="--", alpha=0.7,
                label=f"Best Bio-only R²={bio_best:.3f}")
if not np.isnan(fpc_best):
    ax2.axvline(fpc_best, color="#d62728", lw=1.2, ls="--", alpha=0.7,
                label=f"Best fPC R²={fpc_best:.3f}")
if not np.isnan(comb_best):
    ax2.axvline(comb_best, color="#2ca02c", lw=1.2, ls="--", alpha=0.7,
                label=f"Best Combined R²={comb_best:.3f}")

ax2.axvline(0, color="black", lw=1.2)
ax2.text(min(r2_vals) + 0.01, len(labels) - 0.6,
         "← Negative R² = worse than mean prediction (overfitting)",
         fontsize=8, color="#d62728", style="italic", va="top")

ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels, fontsize=8.5)
ax2.invert_yaxis()
ax2.set_xlabel("LOO-CV R²  [target: peak velocity, m/s]", fontsize=11)
ax2.set_title(f"All {len(res_df)} Models Ranked by LOO-CV R²\n"
              f"Target: Peak Velocity (m/s)  ·  n={n_subj} subjects",
              fontweight="bold", fontsize=13)
ax2.grid(alpha=0.25, axis="x")
ax2.set_xlim(min(r2_vals) - 0.08, max(r2_vals) + 0.12)

# Legend
legend_patches = [
    mpatches.Patch(facecolor="#1f77b4", label=f"Bio Features (B) — {X_B.shape[1]} scalars"),
    mpatches.Patch(facecolor="#d62728", label=f"fPC Full Set (A) — {X_A.shape[1]} PCs"),
    mpatches.Patch(facecolor="#e8a820", label=f"fPC {len(RETAINED_PCS)}-PC Stepwise (A)"),
    mpatches.Patch(facecolor="#2ca02c", label=f"Combined (C) — {X_C.shape[1]} features"),
    mpatches.Patch(facecolor="#888888", label=f"Raw kinematics — {dataset.shape[1]} pts"),
]
ax2.legend(handles=legend_patches, loc="lower right", fontsize=8.5,
           framealpha=0.9, edgecolor="#cccccc")

fig2.tight_layout()
out2 = FIG_DIR / "ml_model_ranking.png"
fig2.savefig(str(out2), dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close(fig2)

# ── 11. Summary print ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
best = res_df.iloc[0]
print(f"\nBEST MODEL:  {best['Model']}  [{best['Features']}]")
print(f"  LOO-CV R² = {best['R2_LOO']:.4f}")
print(f"  RMSE      = {best['RMSE']:.4f} m/s")
print(f"  Overfit   = {best['Overfit_Gap']:.4f}")
print(f"\nStepwise baseline [{len(RETAINED_PCS)} PCs]:  "
      f"R²_LOO = {_lookup(res_df, 'Stepwise OLS', f'fPC ({len(RETAINED_PCS)} PCs)'):.4f}")
print(f"Best Bio-only model:  R²_LOO = {bio_best:.4f}")
print(f"Best fPC model:       R²_LOO = {fpc_best:.4f}")
print(f"\nFigures saved to: {FIG_DIR}")
