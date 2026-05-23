# ML in Sprinting: Predicting Sprint Speed from Whole-Body Kinematics

> *Widely regarded as the most desired physical quality in ball-sports, speed is the trait that separates elite performers from good ones — that creates plays others cannot make.*

---

## Overview

This project applies **functional Principal Component Analysis (fPCA)** and a full suite of machine learning models to whole-body kinematic data from 30 competitive sprinters, separately analysing the **acceleration** (0–20 m) and **top-speed** (≥ 25 m) phases of the 60 m sprint.

**Two headline findings:**

1. **The race is decided in the first 20 metres.** Split times across 0–10 m and 10–20 m predict overall 60 m time better than any kinematic feature in the body (R² up to 0.93–0.99 for early-split → cumulative time vs. R² ≈ 0.46 for the best kinematic model).
2. **Sex differences are about maintenance, not mechanics.** Males and females share the same kinematic template at top speed; the only statistically significant biomechanical difference between sexes is *how long* they sustain peak velocity — females hold top speed ~5 m longer than males on average.

---

## The Problem

Contemporary player-tracking systems reduce speed to single-value summaries — peak velocity, average acceleration — collapsing the temporal structure of movement into a number. This obscures a more fundamental question: **what actually constitutes speed in competitive settings?**

Shohei Ohtani's base-stealing illustrates this precisely. In 2025, his Statcast sprint speed ranked ~72nd percentile in MLB (~28.0 ft/sec), yet his stolen base efficiency exceeded 93% in 2024. His advantage lies not in raw speed, but in **how rapidly and consistently he expresses usable speed within constrained windows of time**.

Existing research examines acceleration or maximum velocity in isolation using linear models to relate discrete joint kinematics to speed. This treats sprint phases as independent phenomena rather than continuous transitions, and assumes linear relationships — even though stride frequency exhibits a logarithmic correlation by the third step (Weyand et al., 2000, 2010). No prior study has modeled sprint velocity continuously across all phases while testing for nonlinear kinematic interactions.

---

## Study Design

**30 OUA / USports-level sprinters** (15M / 15F) completed maximal-effort 60-metre sprints from blocks, instrumented with a **64-marker inertial measurement system** capturing whole-body three-dimensional kinematics at 60 Hz.

| Dimension | Detail |
|-----------|--------|
| Participants | 30 (15 Male, 15 Female) |
| Competition level | OUA / USports |
| Sprint distance | 60 metres (block start) |
| Capture rate | 60 Hz |
| Markers | 64 anatomical landmarks |
| Raw feature space | 19,392 per participant (64 markers × 3 axes × 101 frames) |
| Peak velocity range | 7.00 → 10.02 m/s (Males: 9.25 ± 0.41; Females: 8.01 ± 0.43) |

---

## Pipeline — Three Notebooks

The analysis is split into three reproducible notebooks in `notebooks/`:

| # | Notebook | Purpose |
|---|----------|---------|
| **1** | [`01_Sprint_Ensemble_Viz.ipynb`](notebooks/01_Sprint_Ensemble_Viz.ipynb) | **Skeleton viewer + QA.** Loads every `.c3d` file, GCS-aligns each trial, renders SCR (single-frame) and MCR (multi-frame envelope) views for visual quality control of all 30 participants. |
| **2** | [`02_Kinematics_PCA.ipynb`](notebooks/02_Kinematics_PCA.ipynb) | **C3D processing → stride vectors → fPCA.** Detects sprint start, identifies five strides around peak velocity, time-normalises each stride to 101 frames, height-normalises across participants, builds a 30 × 19,392 matrix, runs PCA. |
| **3** | [`03_Sprint_ML_Analysis.ipynb`](notebooks/03_Sprint_ML_Analysis.ipynb) | **Modelling + interpretation.** Loads pre-processed stride vectors, runs functional PCA, biomechanical feature engineering, 26-model comparison with LOO-CV, feature importance (Ridge / SHAP / permutation), split-time prediction, sex analysis, and PC shape-mode figures. |

A standalone Python utility, [`sprint_animation.py`](sprint_animation.py), renders 4-panel MP4 visualisations (thoracic velocity curve + sagittal skeleton + top-down track view) for any participant. Examples are in `outputs/animations/`.

---

## Sprint Animations

Three representative trials are bundled with the repository (full set of 31 is local-only; see `sprint_animation.py` to regenerate):

| Trial | Sex | Peak velocity | File |
|-------|-----|---------------|------|
| 🏆 **Fastest** | M | **10.02 m/s** | [`outputs/animations/SB101_fastest_M_10.02ms.mp4`](outputs/animations/SB101_fastest_M_10.02ms.mp4) |
| 📊 **Average** | F | **8.62 m/s** | [`outputs/animations/SB16_average_F_8.62ms.mp4`](outputs/animations/SB16_average_F_8.62ms.mp4) |
| 🐢 **Slowest** | F | **7.00 m/s** | [`outputs/animations/SB202_slowest_F_7.00ms.mp4`](outputs/animations/SB202_slowest_F_7.00ms.mp4) |

Each clip starts automatically at the moment the sprinter clears the blocks (idle frames trimmed via thoracic-velocity threshold) and ends at the 62.5 m finish line. Rendered at 20 fps, H.264.

---

## Key Takeaways

### ① Acceleration Phase (0–20 m)

**Modelling.** Kinematic models for acceleration plateaued at a modest ceiling — best LOO-CV R² = **0.186** (LASSO on accelerometer fPC scores, 0–10 m). Ridge, Random Forest, kNN, SVR, and XGBoost all returned near-zero or negative LOO-CV R² for predicting split-time from kinematics alone.

**What the data says faster acceleration *looks like*:**
- **Greater maintanenace of forward trunk lean from Blocks to Step 8 while maintaining forward knee drive** 
- **Greater arm swing ROM**

**Phase descriptives:**
- Distance to reach 95 % of peak velocity: **21.96 ± 5.58 m** (range 14.89 → 42.79)
- Strides to peak velocity: **12.6 ± 4.5** (range 8 → 25)
- **Stride efficiency** (distance ÷ ground-contact time) **roughly doubles between strides 1 and 2** — most of the mechanical optimisation happens in the first 10 m.

**Practical insight.** Acceleration kinematics do **not** generalise tightly across athletes; multiple individual strategies can produce fast acceleration. The signal in the *outcome* (split time) is strong, but the path through the body is heterogeneous.

---

### ② Top-Speed Phase (≥ 25 m)

**Modelling.** Top-speed kinematics are substantially more predictable than acceleration kinematics.

| Rank | Model | Features | LOO-CV R² | RMSE (m/s) | Overfit gap |
|------|-------|----------|-----------|------------|-------------|
| 1 | **SVR (C = 10)** | **9 Bio features** | **0.465** | **0.540** | 0.518 |
| 2 | PLS (n = 6) | Raw 19,392 | 0.421 | 0.562 | 0.427 |
| 3 | PCR (n = 12) | Raw 19,392 | 0.390 | 0.577 | 0.386 |
| 4 | **Stepwise OLS** | **3 PCs (A\*)** | **0.386** | **0.579** | **0.138** ⭐ most generalisable |
| 5 | XGBoost | 9 Bio features | 0.294 | 0.620 | 0.621 |

**Top predictors (convergent across Ridge, RF permutation, SHAP):**
1. **Top-speed maintenance distance** — *longer* = faster
2. **Stride length at top speed** — *longer* = faster
3. **Stride frequency at top speed** — opposing direction (the classic length-vs-frequency trade-off)

**What the data says faster top-end *looks like*:**
- **More upright trunk**
- **Forward foot-strike** 
- **Lower CoM at full support** 
- **Greater hip extension and plantarflexion at toe-off** (ankle-dominant propulsion)
- **Higher / more forward knee** at max vertical projection
- **Compact, near-vertical shank at touchdown** (minimises horizontal braking)

---

### ③ The "30-Metre Barrier"

The strongest predictive signal in the entire dataset is **not** kinematic — it is **early split-time itself**.

| Predictor | → Target | R² (LOO-CV) |
|-----------|----------|------------|
| 0–10 m split | 30 m time | **0.97** |
| 0–10 m split | 60 m time | **0.88** |
| 10–20 m split | 30 m time | **0.99** |
| 10–20 m split | 60 m time | **0.93** |
| Top-speed bio features | 30 m time | ~0.51 |
| Top-speed bio features | 60 m time | ~0.48 |

> **Implication.** The competitive outcome of the 60 m is largely decided within the first 10–20 metres. For coaches working with team-sport athletes (baseball baserunning, soccer breakaway runs, basketball transition), this means **start-acceleration training will return more on the dollar than top-speed work** for distances under ~30 m.

---

### ④ Sex Differences — Maintenance, Not Mechanics

| Metric | Males (mean ± SD) | Females (mean ± SD) | p | Significant |
|--------|-------------------|---------------------|---|-------------|
| **Top-speed maintenance distance** | **32.21 ± 5.90 m** | **37.51 ± 4.62 m** | **0.010** | ✅ **Yes** |
| Stride length at top speed | 3.73 ± 0.69 m | 3.73 ± 0.32 m | 0.41 | No |
| Stride frequency at top speed | 1.82 ± 0.23 Hz | 1.74 ± 0.21 Hz | 0.24 | No |
| Vertical oscillation at top speed | 0.060 ± 0.016 m | 0.056 ± 0.013 m | 0.37 | No |
| Strides to peak velocity | 10.87 ± 1.54 | 14.33 ± 5.52 | 0.11 | No |

The only kinematic metric that significantly separates the two groups is **how long they sustain peak velocity** — and females, surprisingly, hold top-speed *longer* than males (rank-biserial effect size = 0.556). The mechanical template is shared; the sustaining capacity differs.

---

## Block Start Kinematics — Ralph Mann Angle Targets

Ralph Mann's published sprint mechanics provide concrete joint-angle targets for the block-clearance and first two steps. The kinograms below overlay the slowest, median, and fastest sprinter in each sex at five canonical events, with the Mann target annotated beneath each panel:

| Event | Target | Direction |
|-------|--------|-----------|
| Rear Foot Clearance | Rear lower-leg ≈ **145°** | Less extension is better (foot still cocked) |
| Rear Ankle Cross | Rear lower-leg ≈ **87°** | More extension is better (drive completed) |
| Front Foot Clearance | Front lower-leg ≈ **169°**, trunk ≈ **30°** | Stay low — complete extension, minimal trunk lift |
| Step 1 TD | CoG behind front foot at contact | Back straight for effective push-off |
| Step 2 TD | Increased hip height vs Step 1 | Complete knee extension at push-off |

Faster sprinters consistently sit closer to the Mann targets at each frame (e.g. more complete rear-leg drive at Ankle Cross, lower trunk at Front Foot Clearance).

### Males — Block Clearance + First 2 Steps
![Block SCR Males](outputs/figures/kinogram_block_scr_M.png)

### Females — Block Clearance + First 2 Steps
![Block SCR Females](outputs/figures/kinogram_block_scr_F.png)

*Code: `notebooks/02_Kinematics_PCA.ipynb`, cells 15–17 (Block Clearance SCR + First 2 Steps).*

---

## Figures

### Velocity–Distance Profiles, All 30 Participants
Stars mark peak velocity; shading marks the top-speed maintenance window.

![Velocity Curves](outputs/figures/velocity_curves.png)

---

### Velocity Curves by Sex
![Velocity Curves by Sex](outputs/figures/velocity_curves_by_sex.png)

---

### Scree Plot — Variance Explained by PC
![Scree Plot](outputs/figures/scree_plot.png)

---

### Top-Speed Phase — Strongest PC (PC2, r = +0.483 with peak velocity)
![Top-Speed PC2](outputs/figures/top-speed_fpca_PC_2.png)

---

### Acceleration Phase — Strongest PC (PC5, r = −0.486 with peak velocity)
![Accel PC5](outputs/figures/accel_fpca_PC_5.png)

---

### Acceleration fPCA by Step Range — Fast / Median / Slow Overlay

Running fPCA **separately for early acceleration (Steps 3-8) and late acceleration / transition (Steps 9-16)** highlights *where* in the acceleration phase the kinematic signal for speed is strongest. Each figure overlays the slowest, median, and fastest sprinter in the cohort at three within-stride phase points (Touchdown / Mid-Stance / Toe-Off) in both sagittal and frontal views.

| Step range | Top velocity-correlated PC | r with peak velocity (n = 30) |
|------------|----------------------------|------------------------------:|
| **Steps 3-8** (early acceleration) | PC5 | **r = +0.53** |
| **Steps 9-16** (late acceleration / transition) | PC1 | **r = +0.50** |

#### Steps 3-8 — All Participants
![Accel fPCA Steps 3-8 All](outputs/figures/accel_fpca_steps_3-8_all.png)

#### Steps 9-16 — All Participants
![Accel fPCA Steps 9-16 All](outputs/figures/accel_fpca_steps_9-16_all.png)

#### PC1 Shape Mode — Always Shown

PC1 captures the dominant variance in each step range — **94.0%** for Steps 3-8 and **97.7%** for Steps 9-16. Even where PC1 is not the *strongest* velocity-correlated component (see Steps 3-8 above), it is the largest source of inter-athlete kinematic variation and worth visualising directly. The ±2 SD reconstructions below show the full 101-frame stride as a motion envelope, with the bold skeleton at mid-stance for reference.

##### Steps 3-8 — PC1 (94.0% var, r = +0.41 with peak velocity)
![Accel fPCA Steps 3-8 PC1](outputs/figures/accel_fpca_steps_3-8_PC1.png)

##### Steps 9-16 — PC1 (97.7% var, r = +0.50 with peak velocity)
![Accel fPCA Steps 9-16 PC1](outputs/figures/accel_fpca_steps_9-16_PC1.png)

Male-only and female-only variants of each step range, plus per-range scree + correlation plots, are saved as:
- `outputs/figures/accel_fpca_steps_3-8_{M,F,all}.png`
- `outputs/figures/accel_fpca_steps_9-16_{M,F,all}.png`
- `outputs/figures/accel_fpca_steps_{3-8,9-16}_scree.png`

*Code: `notebooks/02_Kinematics_PCA.ipynb`, cell "Acceleration fPCA per step range".*

---

### Model Comparison — LOO-CV R² Across All Algorithms
![Model Comparison](outputs/figures/model_comparison.png)

---

### Overfitting Diagnostic
Train R² (blue) vs. LOO-CV R² (red) as more PCs are added.

![Overfitting Diagnostics](outputs/figures/overfitting_diagnostics.png)

---

### SHAP Feature Importance (XGBoost, Top-Speed Bio Features)
![SHAP Summary](outputs/figures/shap_summary.png)

---

### The "30-m Barrier" — Best-Interval Prediction of Cumulative Time
![Best Interval Prediction](outputs/figures/best_interval_prediction.png)

---

### Cross-Phase R² Comparison
![Cross-Phase R²](outputs/figures/cross_phase_r2_comparison.png)

---

### Sex Boxplots
![Sex Boxplots](outputs/figures/sex_boxplots.png)

---

### Split Times by Sex
![Split Times by Sex](outputs/figures/split_times_by_sex.png)

---

## Repository Structure

```
ML-Biomechanics-Comprehensive-Analysis-of-the-60m-Sprint/
├── README.md                                ← this file
├── .gitignore
├── sprint_animation.py                      ← 4-panel MP4 renderer
├── notebooks/
│   ├── 01_Sprint_Ensemble_Viz.ipynb         ← skeleton viewer + QA
│   ├── 02_Kinematics_PCA.ipynb              ← C3D → stride vectors → fPCA
│   └── 03_Sprint_ML_Analysis.ipynb          ← models + interpretation
└── outputs/
    ├── animations/
    │   ├── SB101_fastest_M_10.02ms.mp4
    │   ├── SB16_average_F_8.62ms.mp4
    │   └── SB202_slowest_F_7.00ms.mp4
    ├── data/
    │   ├── metadata.csv                      ← participant ID, sex, velocity, height
    │   ├── sprint_biomechanics_metrics.csv   ← 18 bio metrics × 30 participants
    │   ├── model_comparison.csv              ← top-speed LOO-CV results
    │   ├── accel_model_comparison.csv        ← acceleration LOO-CV results
    │   ├── split_times.csv                   ← radar-measured split times
    │   ├── sex_differences.csv               ← M / F comparison (Mann-Whitney U)
    │   ├── cross_phase_comparison.csv        ← acceleration vs top-speed
    │   └── phase_prediction_heatmap*.csv     ← interval → target time R² grid
    └── figures/                              ← 30+ publication-ready PNGs
```

---

## Reproduction

```bash
# 1 — install dependencies
pip install numpy pandas scikit-learn scipy matplotlib tqdm \
            statsmodels xgboost shap ezc3d python-docx

# 2 — place raw .c3d files in:
#     ../60m Data/All Sprint Trials/   (not included in repo — participant privacy)

# 3 — run notebooks in order
jupyter notebook notebooks/01_Sprint_Ensemble_Viz.ipynb     # QA / visualisation
jupyter notebook notebooks/02_Kinematics_PCA.ipynb          # stride vectors → fPCA
jupyter notebook notebooks/03_Sprint_ML_Analysis.ipynb      # models + interpretation

# 4 — (optional) render animation for a single trial
python sprint_animation.py SB101              # one trial
python sprint_animation.py --all --fps 20     # batch all 31
```

> Raw `.c3d` and `.xlsx` files are not included for participant-privacy reasons. Contact the author for data access.

---

## References

Beaudette, S. M., Zwambag, D. P., Graham, R. B., & Brown, S. H. M. (2019). Discriminating spatiotemporal movement strategies during spine flexion-extension in healthy individuals. *The Spine Journal*, 19(7), 1264–1275. https://doi.org/10.1016/j.spinee.2019.02.002

Braunstein, B., Goldmann, J.-P., Albracht, K., Sanno, M., Willwacher, S., Heinrich, K., Herrmann, V., & Brüggemann, G.-P. (2013). Joint specific contribution of mechanical power and work during acceleration and top speed in elite sprinters. *ISBS — Conference Proceedings Archive*.

Brazil, A., Exell, T., Wilson, C., Willwacher, S., Bezodis, I., & Irwin, G. (2017). Lower limb joint kinetics in the starting blocks and first stance in athletic sprinting. *Journal of Sports Sciences*, 35(16), 1629–1635. https://doi.org/10.1080/02640414.2016.1227465

Brazil, A., Exell, T., Wilson, C., Willwacher, S., Bezodis, I. N., & Irwin, G. (2018). Joint kinetic determinants of starting block performance in athletic sprinting. *Journal of Sports Sciences*, 36(14), 1656–1662. https://doi.org/10.1080/02640414.2017.1409608

Čoh, M., Hébert-Losier, K., Štuhec, S., Babić, V., & Supej, M. (2018). Kinematics of Usain Bolt's maximal sprint velocity. *Kinesiology*, 50(2), 172–180. https://doi.org/10.26582/k.50.2.10

Colyer, S. L., Nagahara, R., Takai, Y., & Salo, A. I. T. (2018). How sprinters accelerate beyond the velocity plateau of soccer players: Waveform analysis of ground reaction forces. *Scandinavian Journal of Medicine & Science in Sports*, 28(12), 2527–2535. https://doi.org/10.1111/sms.13302

Debaere, S., Delecluse, C., Aerenhouts, D., Hagman, F., & Jonkers, I. (2013). From block clearance to sprint running: Characteristics underlying an effective transition. *Journal of Sports Sciences*, 31(2), 137–149. https://doi.org/10.1080/02640414.2012.722225

Debaere, S., Delecluse, C., Aerenhouts, D., Hagman, F., & Jonkers, I. (2015). Control of propulsion and body lift during the first two stances of sprint running: A simulation study. *Journal of Sports Sciences*, 33(19), 2016–2024. https://doi.org/10.1080/02640414.2015.1026375

Higashihara, A., Nagano, Y., Ono, T., & Fukubayashi, T. (2018). Differences in hamstring activation characteristics between the acceleration and maximum-speed phases of sprinting. *Journal of Sports Sciences*, 36(12), 1313–1318. https://doi.org/10.1080/02640414.2017.1375548

Kariyama, Y., & Zushi, K. (2016). Relationships between lower-limb joint kinetic parameters of sprint running and rebound jump during the support phases. *The Journal of Physical Fitness and Sports Medicine*, 5(2), 187–193. https://doi.org/10.7600/jpfsm.5.187

Morin, J.-B., Bourdin, M., Edouard, P., Peyrot, N., Samozino, P., & Lacour, J.-R. (2012). Mechanical determinants of 100-m sprint running performance. *European Journal of Applied Physiology*, 112(11), 3921–3930. https://doi.org/10.1007/s00421-012-2379-8

Morin, J.-B., Edouard, P., & Samozino, P. (2011). Technical ability of force application as a determinant factor of sprint performance. *Medicine and Science in Sports and Exercise*, 43(9), 1680–1688. https://doi.org/10.1249/MSS.0b013e318216ea37

Rabita, G., Dorel, S., Slawinski, J., Sàez-de-Villarreal, E., Couturier, A., Samozino, P., & Morin, J.-B. (2015). Sprint mechanics in world-class athletes: A new insight into the limits of human locomotion. *Scandinavian Journal of Medicine & Science in Sports*, 25(5), 583–594. https://doi.org/10.1111/sms.12389

Sado, N., Yoshioka, S., & Fukashiro, S. (2020). Three-dimensional kinetic function of the lumbo-pelvic-hip complex during block start. *PloS One*, 15(3), e0230145. https://doi.org/10.1371/journal.pone.0230145

Schache, A. G., Blanch, P. D., Dorn, T. W., Brown, N. A. T., Rosemond, D., & Pandy, M. G. (2011). Effect of running speed on lower limb joint kinetics. *Medicine and Science in Sports and Exercise*, 43(7), 1260–1271. https://doi.org/10.1249/MSS.0b013e3182084929

Slawinski, J., Bonnefoy, A., Levêque, J.-M., Ontanon, G., Riquet, A., Dumas, R., & Chèze, L. (2010). Kinematic and kinetic comparisons of elite and well-trained sprinters during sprint start. *Journal of Strength and Conditioning Research*, 24(4), 896–905. https://doi.org/10.1519/JSC.0b013e3181ad3448

Valamatos, M. J., Abrantes, J. M., Carnide, F., Valamatos, M.-J., & Monteiro, C. P. (2022). Biomechanical performance factors in the track and field sprint start: A systematic review. *International Journal of Environmental Research and Public Health*, 19(7), 4074. https://doi.org/10.3390/ijerph19074074

Vellucci, C. L., & Beaudette, S. M. (2022). A need for speed: Objectively identifying full-body kinematic and neuromuscular features associated with faster sprint velocities. *Frontiers in Sports and Active Living*, 4, 1094163. https://doi.org/10.3389/fspor.2022.1094163

von Lieres Und Wilkau, H. C., Irwin, G., Bezodis, N. E., Simpson, S., & Bezodis, I. N. (2020). Phase analysis in maximal sprinting: An investigation of step-to-step technical changes between the initial acceleration, transition and maximal velocity phases. *Sports Biomechanics*, 19(2), 141–156. https://doi.org/10.1080/14763141.2018.1473479

Werkhausen, A., Willwacher, S., & Albracht, K. (2021). Medial gastrocnemius muscle fascicles shorten throughout stance during sprint acceleration. *Scandinavian Journal of Medicine & Science in Sports*, 31(7), 1471–1480. https://doi.org/10.1111/sms.13956

Weyand, P. G., Sandell, R. F., Prime, D. N. L., & Bundle, M. W. (2010). The biological limits to running speed are imposed from the ground up. *Journal of Applied Physiology*, 108(4), 950–961. https://doi.org/10.1152/japplphysiol.00947.2009

Yada, K., Ae, M., Tanigawa, S., Ito, A., Fukuda, K., & Kijima, K. (2011). Standard motion of sprint running for male elite and student sprinters. *ISBS — Conference Proceedings Archive*.

---

*Analysis pipeline: Python 3.13 · scikit-learn 1.6 · XGBoost 3.0 · SHAP 0.50 · ezc3d · matplotlib · ffmpeg*
