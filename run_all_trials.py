"""
Batch runner: Apply sprint_preprocessing_pipeline to all xlsx trials.

Input:  /60m Data/All Sprint Trials/Sprint Trials in xlsx/
Output: /Shun's Sprints Code/Realigned PCA Sprint Trials/
"""

import sys
from pathlib import Path

# Add the script directory to path so we can import the pipeline
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd
from datetime import datetime
from sprint_preprocessing_pipeline import (
    setup_logging,
    process_trial,
    SEGMENTS,
)

# ── paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/Users/shunchen/Desktop/60m Project Folder")
INPUT_DIR = PROJECT_ROOT / "60m Data" / "All Sprint Trials" / "Sprint Trials in xlsx"
OUTPUT_DIR = SCRIPT_DIR / "Realigned PCA Sprint Trials"

# ── main ───────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(OUTPUT_DIR)

    logger.info("=" * 60)
    logger.info("BATCH SPRINT PREPROCESSING PIPELINE")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input directory:  {INPUT_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("=" * 60)

    # Discover all xlsx files
    xlsx_files = sorted(INPUT_DIR.glob("*.xlsx"))
    if not xlsx_files:
        logger.error("No .xlsx files found in input directory!")
        return

    # Build trial list (same dict format that process_trial expects)
    trials = []
    for fp in xlsx_files:
        stem = fp.stem
        # Parse participant ID: strip trial suffix like -001
        parts = stem.split("-")
        pid = parts[0].strip()
        trials.append({
            "participant_id": pid,
            "filepath": str(fp),
            "filetype": ".xlsx",
        })

    logger.info(f"Found {len(trials)} xlsx trial files:")
    for t in trials:
        logger.info(f"  {t['participant_id']}: {t['filepath']}")

    # Process each trial
    results = []
    for i, trial_info in enumerate(trials):
        pid = trial_info["participant_id"]
        logger.info(f"\n>>> Processing {i + 1}/{len(trials)}: {pid}")
        result = process_trial(trial_info)
        if result is not None:
            results.append(result)
            logger.info(f">>> SUCCESS: {pid}")
        else:
            logger.warning(f">>> FAILED: {pid}")

    # Build output matrix
    if not results:
        logger.error("No participants were successfully processed!")
        return

    n_features = len(results[0]["vector"])
    dataset_matrix = np.zeros((len(results), n_features))
    for i, r in enumerate(results):
        dataset_matrix[i, :] = r["vector"]

    # ── Save outputs ──────────────────────────────────────────────────
    # 1. Participant vectors (.npy)
    npy_path = OUTPUT_DIR / "participant_vectors.npy"
    np.save(npy_path, dataset_matrix)
    logger.info(f"\nSaved participant_vectors.npy: shape={dataset_matrix.shape}")

    # 2. Metadata CSV
    metadata_rows = []
    for r in results:
        metadata_rows.append({
            "participant_id": r["participant_id"],
            "max_velocity_ms": r["max_velocity"],
            "height_m": r["height"],
            "n_strides": r["n_strides"],
            "filepath": r["filepath"],
        })
    metadata_df = pd.DataFrame(metadata_rows)
    csv_path = OUTPUT_DIR / "metadata.csv"
    metadata_df.to_csv(csv_path, index=False)
    logger.info(f"Saved metadata.csv: {len(metadata_df)} participants")

    # ── Summary ───────────────────────────────────────────────────────
    logger.info(f"\n{'=' * 60}")
    logger.info("PIPELINE COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Participants processed: {len(results)}/{len(trials)}")
    logger.info(f"Feature vector size: {n_features}")
    logger.info(f"Dataset shape: {dataset_matrix.shape}")
    logger.info(f"\nOutput files:")
    logger.info(f"  {npy_path}")
    logger.info(f"  {csv_path}")
    logger.info(f"  {OUTPUT_DIR / 'logs.txt'}")

    logger.info(f"\nPer-participant summary:")
    for r in results:
        logger.info(
            f"  {r['participant_id']}: "
            f"v_max={r['max_velocity']:.2f} m/s, "
            f"height={r['height']:.3f} m, "
            f"strides={r['n_strides']}"
        )


if __name__ == "__main__":
    main()
