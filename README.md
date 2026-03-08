# ML in Sprinting: Predicting Sprint Speed from Whole-Body Kinematics

> *Widely regarded as the most desired physical quality in ball-sports, speed is the trait that separates elite performers from good ones — that creates plays others cannot make.*

---

## The Problem

Contemporary player-tracking systems reduce speed to single-value summaries — peak velocity, average acceleration — collapsing the temporal structure of movement into a number. This obscures a more fundamental question: **what actually constitutes speed in competitive settings?**

Shohei Ohtani's base-stealing illustrates this precisely. In 2025, his Statcast sprint speed ranked ~72nd percentile in MLB (~28.0 ft/sec), yet his stolen base efficiency exceeded 93% in 2024. His advantage lies not in raw speed, but in **how rapidly and consistently he expresses usable speed within constrained windows of time**.

Existing research examines acceleration or maximum velocity in isolation using linear models to relate discrete joint kinematics to speed. This treats sprint phases as independent phenomena rather than continuous transitions, and assumes linear relationships — even though stride frequency exhibits a logarithmic correlation by the third step (Weyand et al., 2000, 2010). No prior study has modeled sprint velocity continuously across all phases while testing for nonlinear kinematic interactions, and no quantitative framework exists for measuring upper-extremity kinematic contributions to speed.

---

## Study Design

**31 OUA/USports-level sprinters** (16M / 15F) completed maximal-effort 60-metre sprints from blocks, instrumented with a **64-marker inertial measurement system** capturing whole-body three-dimensional kinematics at 60 Hz.

| Dimension | Detail |
|-----------|--------|
| Participants | 31 (16 Male, 15 Female) |
| Competition level | OUA / USports |
| Sprint distance | 60 metres (block start) |
| Capture rate | 60 Hz |
| Markers | 64 anatomical landmarks |
| Feature space | 19,392 per participant (64 markers × 3 axes × 101 frames) |

---

## Methodology

### 1 — Data Pipeline
Raw `.c3d` motion capture files are loaded per participant. Each trial undergoes:
- **Sprint start detection** — velocity threshold (1.0 m/s sustained for 5 frames)
- **PCA coordinate alignment** — consistent AP/ML/Vertical axis assignment across participants
- **Origin normalisation** — posterior heel at frame 0 set to origin
- **60 m trim** — data cropped to race distance

### 2 — Stride Representation
Five strides around peak velocity are identified per participant. Each stride is:
- Time-normalised to 101 frames
- T12 vertebra-debiased (removes bulk translation)
- Height-normalised (pelvis-to-head distance)
- Averaged across strides into a single **19,392-element vector** per participant

### 3 — Functional PCA (fPCA)
Principal Component Analysis on the 31 × 19,392 feature matrix reduces whole-body kinematics to a compact set of interpretable movement components. Each PC captures a coordinated pattern of joint motion across the full gait cycle.

### 4 — Biomechanical Feature Engineering
18 hand-crafted metrics are extracted per participant:

| Category | Metrics |
|----------|---------|
| Block exit | Footedness, stride 1 & 2 distance, GCT, efficiency ratio |
| Acceleration | Strides to top speed, distance to 95% peak |
| Top speed | Stride length, stride frequency, GCT, flight time, duty factor |
| Mechanics | Vertical oscillation, vertical RFD, top speed maintenance distance |

### 5 — Model Comparison (12 Models × 2 Feature Sets)
All models evaluated via **Leave-One-Out Cross-Validation** (LOO-CV) — the gold standard at n=31. Two feature sets compared:

- **A — fPC Scores** (30 kinematic components)
- **B — Biomechanical Features** (18 hand-crafted metrics)

| Model | A: fPC Scores | B: Bio Features |
|-------|:---:|:---:|
| Stepwise OLS | ✓ | — |
| OLS | ✓ | ✓ |
| Ridge | ✓ | ✓ |
| LASSO | ✓ | ✓ |
| Elastic Net | ✓ | ✓ |
| PCR | ✓ | — |
| PLS | ✓ | — |
| Random Forest | ✓ | ✓ |
| XGBoost | ✓ | ✓ |
| SVR | ✓ | ✓ |
| kNN | ✓ | ✓ |
| Bayesian Ridge | ✓ | ✓ |

### 6 — Explainability
- **Ridge coefficient analysis** with 200-iteration bootstrap confidence intervals
- **LASSO coefficient path** across regularisation strengths
- **Random Forest permutation importance**
- **XGBoost SHAP values** (feature-level attribution)
- **Bootstrap LASSO stability** (200 iterations — features selected >80% = robust)

### 7 — Kinematic Visualisation (SCR / MCR)
Single and Multi-Component Reconstruction render actual 3D skeleton postures at the 5th, 50th, and 95th percentile of each PC — directly comparing how faster and slower sprinters *look* at each phase of the gait cycle.

---

## Preliminary Results

> N/A — analysis ongoing.

---

## Figures

### Velocity–Distance Profiles
All 31 participants. Stars mark peak velocity; shaded zones show top-speed maintenance windows.

![Velocity Curves](outputs/figures/velocity_curves.png)

---

### Single Component Reconstruction — PC5
Sagittal / Frontal / Transverse projections. Red = 5th percentile (slow), Black = mean, Blue = 95th percentile (fast).

![SCR PC5](outputs/figures/SCR_PC5.png)

---

### Multi-Component Reconstruction (All Retained PCs)
Combined effect of all retained PCs at 5 gait-cycle positions (10% → 100%).

![MCR Figure 6](outputs/figures/MCR_figure6.png)

---

### Model Comparison — LOO-CV R²
21 model × feature-set combinations ranked by out-of-sample performance.

![Model Comparison](outputs/figures/model_comparison.png)

---

### SHAP Feature Importance (XGBoost)
Which biomechanical metrics drive velocity predictions — and in which direction.

![SHAP Summary](outputs/figures/shap_summary.png)

---

## Repository Structure

```
Shun's Sprints Code/
├── sprint_fPCA_pipeline.ipynb   ← Single executable notebook (Kernel → Restart & Run All)
└── outputs/
    ├── data/
    │   ├── participant_vectors.npy        (31 × 19,392 kinematic feature matrix)
    │   ├── metadata.csv                   (participant ID, sex, velocity, height)
    │   ├── sprint_biomechanics_metrics.csv (18 biomechanical metrics per participant)
    │   └── model_comparison.csv           (LOO-CV results, all 21 model combinations)
    ├── figures/                           (23 publication-ready PNGs)
    └── logs/                              (per-run pipeline logs)
```

## Reproduction

```
# 1. Install dependencies
pip install numpy pandas scikit-learn scipy matplotlib tqdm statsmodels xgboost lightgbm shap ezc3d

# 2. Place .c3d files in:
#    ../60m Data/All Sprint Trials/Sprint Trials in c3d/

# 3. Open notebook and run all cells
jupyter notebook sprint_fPCA_pipeline.ipynb
# Kernel → Restart & Run All
# Runtime: ~5 minutes
```

---

## References

Weyand, P. G., Sternlight, D. B., Bellizzi, M. J., & Wright, S. (2000). Faster top running speeds are achieved with greater ground forces not more rapid leg movements. *Journal of Applied Physiology*, 89(5), 1991–1999.

Weyand, P. G., Sandell, R. F., Prime, D. N., & Bundle, M. W. (2010). The biological limits to running speed are imposed from the ground up. *Journal of Applied Physiology*, 108(4), 950–961.

Velluci, C., & Beaudette, S. M. (2023). Functional principal component analysis of whole-body kinematics during sprint acceleration. *[Journal details pending].*

---

*Analysis pipeline: Python 3.13 · scikit-learn 1.6 · XGBoost 3.0 · SHAP 0.50 · ezc3d*
