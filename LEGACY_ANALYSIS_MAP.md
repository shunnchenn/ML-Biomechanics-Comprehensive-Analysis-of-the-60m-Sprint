# Legacy analysis map

`(4) Archive/Shun's Sprints Code` and `(4) Archive/Old Outputs` are historical archives (symlinks remain at the old root names). They are not imported by the current pipeline. Keep them for qualitative figures, earlier reasoning, and provenance; use notebooks 04/05/07 for current numbers.

## Big picture → detail

| Legacy item | Current home | Status |
|---|---|---|
| `notebooks/01_Sprint_Ensemble_Viz.ipynb` | `sprint_pipeline.ipynb`, `02_Animations.ipynb` | Superseded |
| `notebooks/02_Kinematics_PCA.ipynb` | `01_Angle_fPCA.ipynb` | Position-PCA retired; angle fPCA descriptive only |
| `notebooks/03_Sprint_ML_Analysis.ipynb` | `03_Kinematics_ML.ipynb`, `05_Validation_Ledger.ipynb` | Cite 05 |
| `sprint_animation.py` | `sprint_outputs.py` | Superseded |
| `_sprint_model_comparison.py` | `model_ladder`, `nested_cv` | Libraryized |
| `_accel_fpca_step_ranges.py` | `07_Acceleration_Process.ipynb` | Cycle averaging retired; steps remain ordered |
| `_topspeed_fpca_shape.py` | `01_Angle_fPCA.ipynb`, `05_Validation_Ledger.ipynb` | Shape retained; inference re-audited |
| Results & Discussion DOCX files | Current manuscript in `(3)` | Retain prose selectively; recompute numbers |

## Retain

- Qualitative kinograms and phase descriptions.
- Acceleration vs top-speed separation.
- Whole-body visualization and coaching translation where it is clearly descriptive.

## Correct or retire

- SVR R² ≈ 0.47 as the headline: current pre-specified top-speed model is LOO R² = +0.256.
- Exhaustive-search R² = +0.403: nested score = −0.020.
- Selected PCs as generalizable predictors: fPCA is unsupervised and currently descriptive.
- Position fPCA after height normalization: 1.75 m scaling is drawing-only.
- More strides improve reliability: median ICC falls from 0.802 (five) to 0.359 (all).
- Stance ROM findings: outcome-dependent window-length artefact.
- `a0` and `RFmax` as separate findings: they are nearly one-to-one.
- OpenSim ID moments as athlete-specific: body mass and GRF are unavailable.

## Current source hierarchy

1. `05_Validation_Ledger.ipynb` — honest top-speed scoreboard.
2. `07_Acceleration_Process.ipynb` — mass-free acceleration process.
3. `04_Audit_Provenance.ipynb` — recomputed audit quantities.
4. `01_Angle_fPCA.ipynb` — descriptive movement variation.
5. `06_OpenSim_Musculoskeletal.ipynb` — isolated IK/definition experiment.

