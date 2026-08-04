# ML in Sprinting: What Faster Actually Looks Like

> *Widely regarded as the most desired physical quality in ball-sports, speed is the trait that separates elite performers from good ones — that creates plays others cannot make.*

Whole-body kinematics from 30 competitive sprinters over 60 m, analysed on a
**phase-normalised** basis — 0–100% of ground contact, then 0–100% of flight —
so athletes with different contact and flight times are compared at the same
point of the same phase rather than at the same fraction of a stride.

---

## Study design

**30 OUA / USports sprinters** (15 M / 15 F) ran maximal 60 m from blocks in a
64-marker inertial suit capturing whole-body 3-D kinematics at 60 Hz.

| Dimension | Detail |
|-----------|--------|
| Participants | 30 (15 M, 15 F) |
| Competition level | OUA / USports |
| Sprint distance | 60 m, block start |
| Capture rate | 60 Hz |
| Markers | 64 anatomical landmarks |
| Peak velocity range | 7.00 → 10.02 m/s |

---

## How the pipeline works

### Everyone on the same axes

Three things fix the frame, in `sprint/frame.py`:

- **Vertical is the raw Xsens Z axis.** The suit measures gravity directly, so
  vertical is *observed*, not inferred from variance — it cannot rotate into the
  horizontal plane when an athlete's lateral spread exceeds their height.
- **Heading is estimated locally**, from pelvis displacement across the window
  being analysed, not from one axis fitted to the whole 60 m. A global axis is
  tilted by lane drift and by the curved path out of the blocks; a per-step
  heading is immune to both and is sign-unambiguous, because the pelvis always
  advances within a step.
- **Translation and scale are removed** — origin at the support foot at
  touchdown, scale by stature. What is left is a yaw-only Procrustes alignment:
  the only free rotation is the one gravity leaves open.

Steps 1–4 are the exception: heading genuinely rotates during block clearance,
so one heading is fixed across those four steps rather than re-estimated per
step, which would erase that rotation as signal.

### Contact and flight, measured rather than assumed

Touchdown and toe-off use the coordinate rule of Zeni et al. (2008) — touchdown
where the heel is furthest ahead of the pelvis, toe-off where the toe is
furthest behind it — each snapped to the ground-contact run in the foot's own
vertical trace. That snap matters: the coordinate rule places toe-off several
frames late, because a foot that has just left the ground is near-stationary and
keeps falling behind the pelvis after it is airborne.

Every step is then resampled to **20 points of contact + 20 points of flight**.
Twenty and not 101, deliberately: at 60 Hz a top-speed contact is only about 6–7
raw frames, so a denser grid would manufacture resolution the capture rate does
not hold. A phase with fewer than 3 raw frames is emitted as NaN rather than
interpolated — early-acceleration flight can be that short.

Contact and flight **durations are kept as explicit features**. Normalising each
phase is what makes shapes comparable, but timing is a determinant of sprint
speed, not a nuisance to divide out.

### A QA gate on every build

`sprint/events.qa()` checks contact time against [0.07, 0.30] s, duty factor
against [0.15, 0.55], and whether `v ≈ step length × step frequency` closes to
within 10%. Failures are printed per athlete, not buried.

### Three interpretable readouts

At n ≈ 30 the useful output is an effect per feature, not a ranking of
algorithms. `sprint/model.py` fits:

1. **Elastic Net** — LOO-CV R², coefficients on standardised features, so each
   reads as "per +1 SD of this feature, that much change in speed". Percentile
   bootstrap CIs, plus `selected_pct`: how often a feature survived the penalty
   across resamples, which is the honest stability measure at this sample size.
2. **Penalised logistic** — fast vs slow outer tertiles, log-odds per SD.
3. **SHAP** — on the linear model, where it is exact (`β·(x − x̄)`). Tree SHAP at
   this sample size would be noise dressed up as attribution.

Curves are compared with **cluster-based permutation testing**, which marks
*where in the phase* fast and slow athletes differ without testing every phase
point independently.

---

## Running it

```bash
pip install -e ".[dev]"

export SPRINT_C3D_DIR="/path/to/trials"     # only if trials are outside data/c3d/
python -m sprint build                       # c3d -> phase-normalised dataset + QA report
python -m sprint analyse                     # models + figures
pytest                                       # 34 tests, no participant data needed
```

`build` looks in `data/c3d/` by default. Read its QA output before anything else:
it reports contact time, duty factor and whether `v = SL × SF` closes, per
athlete. A trial that fails is a detection problem on that trial, not a finding.

Or run the notebooks, which are thin drivers over the same package:

| Notebook | Purpose |
|---|---|
| [`01_Alignment_and_Event_QA`](notebooks/01_Alignment_and_Event_QA.ipynb) | Frame check, per-trial QA table, skeleton grid |
| [`02_Build_Phase_Dataset`](notebooks/02_Build_Phase_Dataset.ipynb) | C3D → phase-normalised curves and per-step scalars |
| [`03_What_Faster_Looks_Like`](notebooks/03_What_Faster_Looks_Like.ipynb) | Elastic Net / logistic / SHAP + figures |

`sprint_animation.py` still renders the 4-panel MP4s; three examples are in
`outputs/animations/`.

### Outputs

| File | Contents |
|---|---|
| `optimal_kinematics_{accel,topspeed}.png` | Fast vs slow mean ±95% CI per joint over contact\|flight, significant clusters shaded |
| `kinogram_{accel,topspeed}.png` | Mean fast and slow pose at fixed points of contact and flight |
| `coefficients_{accel,topspeed}.png` | Elastic Net β with bootstrap CIs, beside fast-vs-slow log-odds |
| `interpretation_{accel,topspeed}.csv` | β, CI, selection %, log-odds and mean \|SHAP\| per feature |
| `qa_report.csv` | Per-trial contact/duty/velocity checks |

Targets: peak velocity for top speed; the 0–10 m split for acceleration, sign-flipped
so a positive coefficient means *faster* in both phases.

---

## Findings

### The 30-metre barrier — unaffected, and still the strongest signal

The strongest predictive signal in the dataset is not kinematic. It is early
split time itself, measured by radar and entirely independent of the marker
pipeline, so it is untouched by the reprocessing below.

| Predictor | → Target | R² (LOO-CV) |
|-----------|----------|------------|
| 0–10 m split | 30 m time | 0.97 |
| 0–10 m split | 60 m time | 0.88 |
| 10–20 m split | 30 m time | 0.99 |
| 10–20 m split | 60 m time | 0.93 |

> The competitive outcome of a 60 m is largely decided in the first 10–20 m. For
> team-sport athletes covering under ~30 m, start-acceleration work returns more
> than top-speed work.

### Kinematic findings — being regenerated

The kinematic results previously published here were built on a ground-contact
detector that did not work, so they are not reproduced. The defect, visible in
files committed to this repository:

- `sprint_biomechanics_metrics.csv` reports mean top-speed ground contact of
  **0.622 s**, longer than the mean stride period it also reports (0.574 s).
  Contact cannot exceed the whole stride.
- The detector called `find_peaks(distance=30)` on 60 Hz data, but a top-speed
  stride at ~2.3 strides/s is only ~26 frames — the minimum peak spacing was
  wider than the event being detected, so consecutive contacts merged.
  Consistent with that, `accel_participant_vectors.npy` contains **two
  left-heel contacts inside 17 of 26 nominally single strides**.
- `avg_stride_length_top_speed_m × stride_freq_top_speed_hz` = 3.73 × 1.78 =
  **6.6 m/s** against a reported 8.6 m/s peak. Those columns were computed over
  every contact in the trial, not the top-speed window, despite their names.

Anything downstream of those events — the stride vectors, the fPCA modes fitted
to them, the model comparison table, the sex-difference rows for stride length,
stride frequency and vertical oscillation — inherits the error. Regenerating
requires the raw `.c3d` files, which are not in this repository.

`python -m sprint build` now refuses to report these quantities silently: the QA
gate flags any trial whose contact time, duty factor, or `v = SL × SF` identity
falls outside physiological bounds.

### Archived outputs

Everything under `outputs/figures/` and most of `outputs/data/` was produced by
the superseded pipeline. The files are retained for reference and will be
replaced on the next build. `split_times.csv` is radar-measured and remains
valid.

---

## Repository

```
sprint/                 ← the pipeline (~900 lines)
  config.py             marker roles, paths, phase-base constants
  io.py                 c3d loading, marker-role resolution by label
  frame.py              gravity-Z + local heading, yaw-only Procrustes
  events.py             Zeni contacts, step table, phase normalisation, QA
  features.py           sagittal angle curves + per-step scalars
  model.py              Elastic Net, logistic contrast, linear SHAP, SPM
  figures.py            ribbon, coefficient plot, kinogram
  skeleton.py           64-marker bone connectivity (shared with the animator)
  cli.py                python -m sprint build | analyse
tests/                  synthetic-gait fixture + 34 tests
notebooks/              three thin drivers
data/c3d/               raw trials (populated only in the private repo)
sprint_animation.py     4-panel MP4 renderer
outputs/                data, figures, animations
RUNBOOK.md              private-repo + first-real-run checklist
```

Marker indices in `config.py` are confirmed two independent ways against
`outputs/data/accel_participant_vectors.npy`: lateral-coordinate sign, and
membership of the left/right bone lists in `skeleton.py`.

### Data availability

Raw trials are whole-body marker trajectories from 30 identifiable human research
participants. Gait kinematics is a biometric, and in a cohort this small —
30 OUA/USports sprinters, with sex and stature also published — re-identification
is plausible. Two supported configurations:

| Repository | `data/c3d/` | Finding the trials |
|---|---|---|
| **Public** | empty | `export SPRINT_C3D_DIR=/path/to/trials`; data never enters git |
| **Private** | populated | default path, no env var |

Committing trials to a *public* repository cannot be undone — git history, any
existing fork, and GitHub's retention of unreachable commits all outlive a
deletion. `RUNBOOK.md` gives the order of operations that makes the private
configuration safe; `data/README.md` documents the expected layout.

Everything in `outputs/data/` is derived and aggregate — scalars, phase curves,
split times, coefficients — and carries no marker trajectories.

For access to the raw trials, contact the author.

---

## Known limitations

- **Event timing resolution.** Without force plates, touchdown and toe-off are
  estimates. At 60 Hz one frame is 17 ms against a top-speed contact of roughly
  100 ms, so per-step contact time carries about 17% resolution error.
  Averaging over steps reduces it; it does not remove it. Duty-factor results
  should be read with that in mind.
- **Sample size.** n ≈ 30 with more candidate features than athletes. The
  Elastic Net is regularised and cross-validated, and `selected_pct` reports
  stability, but no result here should be treated as a fitted constant.
- **Bootstrap intervals are widths, not exact coverage.** An L1-penalised
  coefficient has a point mass at exactly zero, so its bootstrap distribution is
  not centred on the full-data estimate. The penalty is held fixed at the
  full-data value during resampling (re-tuning it per resample shrinks every
  coefficient and pushes the interval off the estimate entirely), which gives
  full coverage on this dataset — but `selected_pct` remains the more
  trustworthy stability measure.
- **The logistic contrast drops the middle tertile**, so its AUC is not
  comparable to the regression R² — it is an easier problem by construction.

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

Zeni, J. A., Richards, J. G., & Higginson, J. S. (2008). Two simple methods for determining gait events during treadmill and overground walking using kinematic data. *Gait & Posture*, 27(4), 710–714. https://doi.org/10.1016/j.gaitpost.2007.07.007

---

*Python 3.11 · scikit-learn 1.9 · SHAP 0.51 · ezc3d 1.7 · matplotlib · ffmpeg*
