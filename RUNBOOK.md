# Runbook — bring the trials in, then run the pipeline for real

Written for a **local Claude Code session on the Mac**, where
`/Users/shunchen/Desktop/60m Project Folder` is readable. The remote/web session
cannot do any of this: it runs in a Linux container with no access to that disk.

---

## Part 1 — make the repository private, then add the data

**Steps 1 and 2 must complete before step 4. Do not reorder.**

GitHub splits existing public forks into a separate network when a repository is
made private. It does **not** make those forks private, and it does not delete
them. This repository already has one fork. Because the trials are not in the
repository today, that fork cannot receive them — *as long as the visibility flip
happens first*. Reversed, even briefly, the raw participant data is public
permanently.

### 1. Flip the repository to private

GitHub → the repo → Settings → General → Danger Zone → Change visibility →
Make private. Do this by hand; it should be a deliberate act.

### 2. Check the existing fork

```bash
gh api repos/shunnchenn/ML-Biomechanics-Comprehensive-Analysis-of-the-60m-Sprint/forks \
  --jq '.[] | "\(.full_name)  owner=\(.owner.login)  private=\(.private)"'
```

After the flip it becomes a standalone public repo carrying today's code and
figures. No participant data, so this is not urgent — but decide whether to ask
the owner to delete it.

### 3. Pull this branch

```bash
git checkout claude/gait-analysis-refactor-u94wgc
git pull origin claude/gait-analysis-refactor-u94wgc
```

### 4. Un-ignore the data directory

`.gitignore` currently blanket-ignores `*.c3d` and `*.xlsx`. Scope them so stray
captures elsewhere stay ignored but the curated directory is trackable — replace
the "Raw motion-capture data" block with:

```gitignore
# ── Raw motion-capture data ──────────────────────────────────────────────────
# Tracked ONLY in the private repo. See data/README.md before changing this.
*.c3d
*.xlsx
*.xls
!data/c3d/*.c3d
!data/*.xlsx
```

### 5. Measure before choosing a transport

```bash
SRC="/Users/shunchen/Desktop/60m Project Folder/31 Trials Data Folder/C3D, XLSX/Sprint Trials in c3d"
du -sh "$SRC"
ls -lSh "$SRC" | head -3          # largest single file
```

- Largest file **> 50 MB** → use Git LFS: `git lfs install`,
  `git lfs track "data/c3d/*.c3d"`, commit `.gitattributes` **before** the data.
- Otherwise → **plain git**.

Revising the earlier LFS recommendation now that the repo has been measured: the
working tree is 46 MB and history 216 MB, and 30 write-once trials of a few MB
each land well inside GitHub's limits (it warns per-file at 50 MB, blocks at 100
MB, and nags on repos past 1 GB). LFS would add a 1 GB free-tier quota and force
`git lfs install` on every future clone, for no benefit on files that are written
once and never modified.

### 6. Copy the trials in

```bash
mkdir -p data/c3d
cp "$SRC"/*.c3d data/c3d/
ls data/c3d/*.c3d | wc -l          # expect 30 tracked (SB17 is excluded at build time)
```

### 7. Verify staging before committing

```bash
git check-ignore -v data/c3d/*.c3d     # must print NOTHING
git status --short data/c3d | wc -l    # must match the file count
git diff --cached --stat               # nothing unexpected
```

### 8. Confirm private — last reversible moment

```bash
gh repo view shunnchenn/ML-Biomechanics-Comprehensive-Analysis-of-the-60m-Sprint \
  --json visibility --jq .visibility
```

Must print `private`. If it prints `public`, **stop** and go back to step 1.

### 9. Commit and push

```bash
git add data/c3d .gitignore
git commit -m "Add raw C3D trials (private repo only)"
git push origin claude/gait-analysis-refactor-u94wgc
```

---

## Part 2 — first real run

```bash
pip install -e ".[dev]"
python -m sprint build
```

`build` prints the QA gate. **Read this before anything else — it is the point of
the rewrite.** The previous pipeline's contact detector was broken, and these three
checks are what catch it:

| Check | Expect | Old pipeline gave |
|---|---|---|
| `gct_in_range` | top-speed contact ≈ **0.09–0.12 s** | 0.622 s (longer than the stride) |
| `v_matches_sl_x_sf` | closes within 10% | 6.6 m/s vs 8.6 m/s reported peak |
| `duty_in_range` | 0.15–0.55 | — |

Also note any athlete listed under `failing:`. A failing trial is a **detection
problem on that trial**, not a finding — report it rather than widening the bounds
in `sprint/config.py` until it disappears.

Then:

```bash
python -m sprint analyse    # figures + interpretation_{accel,topspeed}.csv
pytest                      # 34 tests
```

Outputs land in `outputs/figures/`:
`optimal_kinematics_{accel,topspeed}.png`, `kinogram_*.png`, `coefficients_*.png`.

---

## Part 3 — the rest of the workspace

Worth inventorying against the repo:

- **`Sprint Blocks Kinogram.xlsx`** (Kinogram Notes sheet) — Ralph Mann joint-angle
  targets. Needed only if the block-start annotations are wanted back, now on the
  contact/flight phase base rather than the deleted ALTIS one.
- **Whatever `.xlsx` produced `outputs/data/split_times.csv`** — worth verifying
  the committed CSV against source. The 30 m barrier is the one finding that
  survived the detector defect, and it rests entirely on these radar splits.
- **`MatLab/`, `Chris V. Files/`** — reference implementations. Useful as an
  independent check on the alignment; both stay gitignored.
- **`(FINAL) Sprint_Ensemble_Viz.ipynb`, `Shun's Sprints Code/`** — superseded by
  the `sprint/` package. Confirm nothing there was dropped in the rewrite.

## Part 4 — regenerate the README numbers

Once `build` passes QA, replace the "Kinematic findings — being regenerated"
section of `README.md` with what the corrected pipeline actually produces. If a
previously published finding does not survive the fix, say so explicitly rather
than quietly dropping it.
