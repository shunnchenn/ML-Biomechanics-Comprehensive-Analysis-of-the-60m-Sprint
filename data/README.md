# data/

## `data/c3d/` — raw trials

Where `python -m sprint build` looks for trials by default
(`sprint/config.py` → `C3D_DIR`, overridable with `SPRINT_C3D_DIR`).

Expected layout — one file per participant, participant ID first:

```
data/c3d/SB061.c3d
data/c3d/SB101.c3d
data/c3d/SB70-001.c3d      # trailing -NNN is fine, the ID is the part before the dash
```

`sprint/cli.py` derives the participant ID as `path.stem.split("-")[0]`, so both
`SB101.c3d` and `SB70-001.c3d` resolve correctly. `SB17` is skipped
(`EXCLUDED_PIDS` — its T8 marker lost tracking).

## Whether this directory is populated

**It is empty in the public repository, and that is deliberate.**

These are whole-body marker trajectories from 30 identifiable human research
participants. Gait kinematics is a biometric — gait recognition is an established
identification modality — and combined with sex, stature and "OUA/USports
sprinter", the cohort is small enough that re-identification is plausible.

Two supported configurations:

| Repository | `data/c3d/` | How the pipeline finds trials |
|---|---|---|
| Public | empty | `export SPRINT_C3D_DIR=/path/to/trials` — data stays off the repo |
| Private | populated | default path, no env var needed |

Committing the trials to a **public** repository is not reversible: git history,
any existing fork, and GitHub's retention of unreachable commits all survive a
later deletion. See `RUNBOOK.md` for the order of operations that makes the
private-repo configuration safe.

## What is safe to commit either way

`outputs/data/` holds derived, aggregate results — per-participant scalars, phase
curves, split times, model coefficients. These carry no marker trajectories and
are fine in a public repository.
