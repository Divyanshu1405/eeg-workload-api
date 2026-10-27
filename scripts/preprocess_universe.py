"""
UNIVERSE Dataset Preprocessing Pipeline CLI.
UNIVERSE: UNobtrusIVE measuRement of mental workload and stress in uncontrolled environments.

This script executes the complete preprocessing and feature extraction workflow:
1. Discovers raw participant session folders (UN_101 to UN_124)
2. Loads Muse S EEG continuous signals (TP9, AF7, AF8, TP10 @ 256 Hz)
3. Evaluates Signal Quality Indices (SQI) and flags artifacts
4. Applies zero-phase Butterworth bandpass (0.5-45 Hz) & 50 Hz notch filter
5. Applies Common Average Referencing (CAR) and normalization
6. Segments into sliding windows (4s with 50% overlap)
7. Synchronizes windows with task ground truth and workload ratings
8. Extracts spectral band powers, theta/alpha workload ratio, engagement index, FAA, Hjorth parameters
9. Creates subject-independent train / val / test splits
10. Saves processed dataset tables and metadata to data/processed/universe/
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from training.preprocessing.dataset_loader import (
    UniverseDatasetLoader,
    UniverseEpochRecord,
    DatasetSplit,
)
from training.preprocessing.universe_preprocessor import (
    UniverseEEGPreprocessor,
    PreprocessingConfig,
)
from training.preprocessing.feature_extraction import (
    UniverseFeatureExtractor,
    FeatureConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("UniversePreprocessor")


def run_pipeline(
    raw_dir: Path,
    output_dir: Path,
    window_sec: float = 4.0,
    overlap_ratio: float = 0.5,
    notch_freq: float = 50.0,
    generate_mock_if_empty: bool = True,
) -> Dict:
    """
    Executes the end-to-end preprocessing pipeline for the UNIVERSE dataset.
    """
    loader = UniverseDatasetLoader(raw_data_dir=raw_dir)
    participants = loader.discover_participants()

    if not participants and generate_mock_if_empty:
        logger.info("No raw participants found in %s. Generating synthetic UNIVERSE mock sessions...", raw_dir)
        mock_subs = ["UN_101", "UN_102", "UN_103"]
        for sub in mock_subs:
            loader.generate_synthetic_universe_session(
                output_dir=raw_dir,
                participant_id=sub,
                session="Lab1",
                duration_seconds=90.0,
            )
            loader.generate_synthetic_universe_session(
                output_dir=raw_dir,
                participant_id=sub,
                session="Wild",
                duration_seconds=60.0,
            )
        participants = loader.discover_participants()

    if not participants:
        logger.error("No participant data found in %s. Exiting.", raw_dir)
        return {"status": "error", "message": "No data found"}

    logger.info("Found %d participants: %s", len(participants), ", ".join(participants))

    # Initialize Preprocessor & Feature Extractor
    prep_cfg = PreprocessingConfig(
        sampling_rate=256.0,
        channels=["TP9", "AF7", "AF8", "TP10"],
        bandpass_low=0.5,
        bandpass_high=45.0,
        notch_freq=notch_freq,
        window_seconds=window_sec,
        overlap_ratio=overlap_ratio,
        apply_car=True,
        normalization="zscore",
        min_quality_score=0.40,
    )
    preprocessor = UniverseEEGPreprocessor(prep_cfg)

    feat_cfg = FeatureConfig(
        sampling_rate=256.0,
        channel_names=["TP9", "AF7", "AF8", "TP10"],
        extract_time_domain=True,
        extract_hjorth=True,
        extract_cognitive_indices=True,
    )
    feature_extractor = UniverseFeatureExtractor(feat_cfg)

    records: List[UniverseEpochRecord] = []
    total_raw_windows = 0
    usable_windows = 0
    rejected_windows = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    for sub_id in participants:
        sessions = loader.discover_sessions(sub_id)
        logger.info("Processing participant %s (sessions: %s)...", sub_id, sessions)

        for session_name in sessions:
            session_dir = raw_dir / sub_id / session_name
            eeg_file = loader.find_eeg_file(session_dir)
            if not eeg_file:
                logger.warning("No EEG file found for %s/%s", sub_id, session_name)
                continue

            labels_file = loader.find_task_labels_file(session_dir)
            task_intervals = loader.read_task_labels(labels_file)

            timestamps, eeg_data, channels = loader.read_eeg_csv(eeg_file)
            if eeg_data.shape[1] == 0:
                logger.warning("EEG data is empty for %s/%s", sub_id, session_name)
                continue

            start_t = float(timestamps[0]) if len(timestamps) > 0 else 0.0

            # Continuous filtering + window segmentation + SQA
            windows = preprocessor.segment_and_preprocess_continuous(
                continuous_eeg=eeg_data,
                channel_names=channels,
                start_timestamp_sec=start_t,
            )

            for win in windows:
                total_raw_windows += 1
                if not win.is_usable:
                    rejected_windows += 1
                    continue

                usable_windows += 1

                # Align window with task ground truth
                task = loader.align_window_with_task(win.start_time_sec, win.end_time_sec, task_intervals)
                if task is not None:
                    workload_lvl = task.workload_level
                    workload_score = task.workload_score
                    task_name = task.task_name
                else:
                    workload_lvl = "unlabeled"
                    workload_score = -1.0
                    task_name = "unknown"

                # Extract spectral, workload, and temporal features
                feats = feature_extractor.extract_features_from_window(win.data, win.channel_names)

                rec = UniverseEpochRecord(
                    participant_id=sub_id,
                    session=session_name,
                    window_index=win.window_index,
                    start_time=win.start_time_sec,
                    end_time=win.end_time_sec,
                    features=feats,
                    workload_level=workload_lvl,
                    workload_score=workload_score,
                    task_name=task_name,
                    is_usable=win.is_usable,
                    quality_score=win.quality_metrics.overall_quality_score,
                )
                records.append(rec)

    logger.info("Preprocessing complete.")
    logger.info("Total windows extracted: %d | Usable: %d | Rejected by SQI: %d (%.1f%%)",
                total_raw_windows, usable_windows, rejected_windows,
                (rejected_windows / max(1, total_raw_windows)) * 100)

    if not records:
        logger.error("No usable epoch records extracted!")
        return {"status": "error", "message": "No usable records"}

    # Generate subject-independent splits
    splits = loader.create_subject_independent_split(participants=participants)
    logger.info("Subject-independent splits:")
    logger.info("  Train (%d): %s", len(splits.train_subjects), splits.train_subjects)
    logger.info("  Val   (%d): %s", len(splits.val_subjects), splits.val_subjects)
    logger.info("  Test  (%d): %s", len(splits.test_subjects), splits.test_subjects)

    # Save CSV files for Train, Val, Test, and All
    feature_names = sorted(list(records[0].features.keys()))
    header = ["participant_id", "session", "window_index", "start_time", "end_time", "task_name", "workload_level", "workload_score", "quality_score"] + feature_names

    def save_records_to_csv(filepath: Path, sub_records: List[UniverseEpochRecord]):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for r in sub_records:
                row = [
                    r.participant_id,
                    r.session,
                    str(r.window_index),
                    f"{r.start_time:.3f}",
                    f"{r.end_time:.3f}",
                    r.task_name,
                    r.workload_level,
                    f"{r.workload_score:.2f}",
                    f"{r.quality_score:.3f}",
                ]
                row.extend([f"{r.features.get(k, 0.0):.6e}" for k in feature_names])
                f.write(",".join(row) + "\n")

    # Save full dataset
    full_csv = output_dir / "universe_features_all.csv"
    save_records_to_csv(full_csv, records)

    train_recs = [r for r in records if r.participant_id in splits.train_subjects]
    val_recs = [r for r in records if r.participant_id in splits.val_subjects]
    test_recs = [r for r in records if r.participant_id in splits.test_subjects]

    save_records_to_csv(output_dir / "universe_features_train.csv", train_recs)
    save_records_to_csv(output_dir / "universe_features_val.csv", val_recs)
    save_records_to_csv(output_dir / "universe_features_test.csv", test_recs)

    # Workload label counts
    label_distribution = {}
    for r in records:
        label_distribution[r.workload_level] = label_distribution.get(r.workload_level, 0) + 1

    metadata = {
        "dataset_name": "UNIVERSE: UNobtrusIVE measuRement of mental workload and stress in uncontrolled environments",
        "hardware": "Muse S (TP9, AF7, AF8, TP10)",
        "sampling_rate": 256.0,
        "filters": {
            "bandpass": [0.5, 45.0],
            "bandpass_order": 4,
            "notch_freq": notch_freq,
        },
        "re_referencing": "Common Average Reference (CAR)",
        "window_duration_seconds": window_sec,
        "window_overlap_ratio": overlap_ratio,
        "feature_count": len(feature_names),
        "total_usable_windows": len(records),
        "total_rejected_windows": rejected_windows,
        "label_distribution": label_distribution,
        "splits": {
            "train_subjects": splits.train_subjects,
            "val_subjects": splits.val_subjects,
            "test_subjects": splits.test_subjects,
            "train_samples": len(train_recs),
            "val_samples": len(val_recs),
            "test_samples": len(test_recs),
        },
    }

    metadata_path = output_dir / "universe_pipeline_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved feature datasets and metadata to %s", output_dir)
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Preprocess the UNIVERSE EEG dataset.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "universe",
        help="Path to raw UNIVERSE dataset folder (containing UN_xxx directories).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "universe",
        help="Path where processed CSV and metadata will be saved.",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=4.0,
        help="Sliding window duration in seconds (default: 4.0s).",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Sliding window overlap ratio (default: 0.5 for 50%).",
    )
    parser.add_argument(
        "--notch-freq",
        type=float,
        default=50.0,
        help="Powerline notch filter frequency in Hz (default: 50.0).",
    )
    parser.add_argument(
        "--no-mock",
        action="store_true",
        help="Do not generate synthetic mock data if raw directory is empty.",
    )

    args = parser.parse_args()
    run_pipeline(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        window_sec=args.window_sec,
        overlap_ratio=args.overlap,
        notch_freq=args.notch_freq,
        generate_mock_if_empty=not args.no_mock,
    )


if __name__ == "__main__":
    main()
