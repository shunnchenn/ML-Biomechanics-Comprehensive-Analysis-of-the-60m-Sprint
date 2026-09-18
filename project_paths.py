"""Portable paths for the 60 m sprint project.

All paths are derived from this file's location. Environment variables can
override roots when data, outputs, manuscripts, or OpenSim live elsewhere.
No directory is created at import time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_path(name: str) -> Path | None:
    """Return an expanded environment path, or None when unset."""
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def _first_existing(*candidates: Path) -> Path:
    """Return the first existing candidate, otherwise the first candidate."""
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _contains_sprint_data(path: Path) -> bool:
    """Return whether a candidate contains the cohort sheet or C3D folder."""
    return (
        (path / "60m Participant Anthropometrics.xlsx").exists()
        or (path / "Sprint Trials in c3d").is_dir()
        or (path / "C3D, XLSX" / "Sprint Trials in c3d").is_dir()
    )


def _data_root() -> Path:
    """Resolve the live data root across the organized and legacy layouts."""
    override = _env_path("SPRINT_DATA_ROOT")
    if override is not None:
        return override
    candidates = (
        ORGANIZED_DATA_ROOT / "31 Trials Data Folder",
        ORGANIZED_DATA_ROOT,
        LEGACY_DATA_ROOT,
    )
    for candidate in candidates:
        if _contains_sprint_data(candidate):
            return candidate.resolve()
    return candidates[0].resolve()


PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = _env_path("SPRINT_PROJECT_ROOT") or PIPELINE_ROOT.parent

RESEARCH_ROOT = _env_path("SPRINT_RESEARCH_ROOT") or (
    PROJECT_ROOT / "(1) Research Resources"
)
ORGANIZED_DATA_ROOT = PROJECT_ROOT / "(2) Data (C,P,A)"
LEGACY_DATA_ROOT = PROJECT_ROOT / "31 Trials Data Folder"
DATA_ROOT = _data_root()

C3D_DIR = _env_path("SPRINT_C3D_DIR") or _first_existing(
    DATA_ROOT / "C3D, XLSX" / "Sprint Trials in c3d",
    DATA_ROOT / "Sprint Trials in c3d",
    LEGACY_DATA_ROOT / "C3D, XLSX" / "Sprint Trials in c3d",
)
ANTHRO_PATH = _env_path("SPRINT_ANTHRO_PATH") or _first_existing(
    DATA_ROOT / "60m Participant Anthropometrics.xlsx",
    LEGACY_DATA_ROOT / "60m Participant Anthropometrics.xlsx",
)

DELIVERABLES_ROOT = _env_path("SPRINT_DELIVERABLES_ROOT") or (
    PROJECT_ROOT / "(3) Manuscripts, Presentations"
)
MANUSCRIPT_DIR = _env_path("SPRINT_MANUSCRIPT_DIR") or (
    DELIVERABLES_ROOT / "Abstracts, Manuscripts"
)
PRESENTATION_DIR = DELIVERABLES_ROOT / "Presentation Slide Deck"
LEGACY_CODE_ROOT = PROJECT_ROOT / "Shun's Sprints Code"
OLD_OUTPUTS_ROOT = PROJECT_ROOT / "Old Outputs"

FIGURES_DIR = _env_path("SPRINT_FIGURES_DIR") or PIPELINE_ROOT / "figures"
MP4_DIR = _env_path("SPRINT_MP4_DIR") or PIPELINE_ROOT / "mp4s"
RESULTS_DIR = _env_path("SPRINT_RESULTS_DIR") or PIPELINE_ROOT / "results"
OPENSIM_DIR = _env_path("SPRINT_OPENSIM_DIR") or PIPELINE_ROOT / "opensim"
OPENSIM_CMD = _env_path("SPRINT_OPENSIM_CMD") or Path(
    "/Applications/OpenSim 4.5/bin/opensim-cmd"
)


@dataclass(frozen=True)
class ProjectPaths:
    """Named project paths shared by scripts and notebooks."""

    pipeline: Path = PIPELINE_ROOT
    project: Path = PROJECT_ROOT
    research: Path = RESEARCH_ROOT
    data: Path = DATA_ROOT
    c3d: Path = C3D_DIR
    anthropometrics: Path = ANTHRO_PATH
    deliverables: Path = DELIVERABLES_ROOT
    manuscripts: Path = MANUSCRIPT_DIR
    presentations: Path = PRESENTATION_DIR
    legacy_code: Path = LEGACY_CODE_ROOT
    old_outputs: Path = OLD_OUTPUTS_ROOT
    figures: Path = FIGURES_DIR
    mp4s: Path = MP4_DIR
    results: Path = RESULTS_DIR
    opensim: Path = OPENSIM_DIR
    opensim_cmd: Path = OPENSIM_CMD

    def missing_inputs(self) -> list[Path]:
        """Return required input paths that do not currently exist."""
        return [
            path
            for path in (self.c3d, self.anthropometrics)
            if not path.exists()
        ]


PATHS = ProjectPaths()
