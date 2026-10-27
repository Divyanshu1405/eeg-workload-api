"""
Automated unit and integration test suite for the UNIVERSE EEG preprocessing pipeline.
"""

from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from training.preprocessing.signal_quality import SignalQualityAssessor
from training.preprocessing.universe_preprocessor import (
    UniverseEEGPreprocessor,
    PreprocessingConfig,
)
from training.preprocessing.feature_extraction import (
    UniverseFeatureExtractor,
    FeatureConfig,
)
from training.preprocessing.dataset_loader import (
    UniverseDatasetLoader,
    DatasetSplit,
)
from app.services.eeg_service import EEGService
from app.schemas.prediction import PredictionRequest


def test_signal_quality_assessor():
    assessor = SignalQualityAssessor(sampling_rate=256.0)
    channels = ["TP9", "AF7", "AF8", "TP10"]
    n_samples = 256 * 4

    # 1. Test clean simulated EEG
    t = np.arange(n_samples) / 256.0
    clean_eeg = np.zeros((4, n_samples))
    for i in range(4):
        clean_eeg[i, :] = 10.0 * np.sin(2 * np.pi * 10.0 * t) + np.random.normal(0, 1.0, n_samples)

    metrics = assessor.evaluate_window(clean_eeg, channels)
    assert metrics.is_window_usable, "Clean EEG should be deemed usable"
    assert metrics.overall_quality_score > 0.5

    # 2. Test flatline channel
    flat_eeg = clean_eeg.copy()
    flat_eeg[1, :] = 0.0  # AF7 flatline
    metrics_flat = assessor.evaluate_window(flat_eeg, channels)
    assert "AF7" in metrics_flat.bad_channel_names
    assert metrics_flat.channel_metrics["AF7"].status == "flatline"

    # 3. Test clipping channel
    clipped_eeg = clean_eeg.copy()
    clipped_eeg[0, :] = clean_eeg[0, :] + 500.0  # TP9 severe clipping with variation
    metrics_clip = assessor.evaluate_window(clipped_eeg, channels)
    assert "TP9" in metrics_clip.bad_channel_names
    assert metrics_clip.channel_metrics["TP9"].status == "clipped"


def test_universe_eeg_preprocessor():
    config = PreprocessingConfig(
        sampling_rate=256.0,
        channels=["TP9", "AF7", "AF8", "TP10"],
        bandpass_low=0.5,
        bandpass_high=45.0,
        notch_freq=50.0,
        apply_car=True,
        normalization="zscore",
        window_seconds=4.0,
        overlap_ratio=0.5,
    )
    preprocessor = UniverseEEGPreprocessor(config)

    # 10 seconds continuous signal with 50 Hz powerline hum and DC offset
    fs = 256.0
    n_samples = int(fs * 10.0)
    t = np.arange(n_samples) / fs

    raw_data = np.zeros((4, n_samples))
    for i in range(4):
        # DC offset + 10 Hz alpha rhythm + 50 Hz noise
        raw_data[i, :] = 50.0 + 15.0 * np.sin(2 * np.pi * 10.0 * t) + 20.0 * np.sin(2 * np.pi * 50.0 * t)

    windows = preprocessor.segment_and_preprocess_continuous(raw_data)
    assert len(windows) > 1, f"Expected multiple sliding windows, got {len(windows)}"

    for win in windows:
        assert win.data.shape == (4, int(fs * 4.0))
        # Zero-mean check from zscore normalization
        for ch in range(4):
            assert abs(np.mean(win.data[ch, :])) < 0.1


def test_universe_feature_extractor():
    extractor = UniverseFeatureExtractor(
        FeatureConfig(sampling_rate=256.0, channel_names=["TP9", "AF7", "AF8", "TP10"])
    )

    n_samples = 256 * 4
    t = np.arange(n_samples) / 256.0
    dummy_eeg = np.zeros((4, n_samples))

    # Give AF7/AF8 strong theta (6 Hz) to simulate high mental workload
    dummy_eeg[1, :] = 20.0 * np.sin(2 * np.pi * 6.0 * t)  # AF7
    dummy_eeg[2, :] = 20.0 * np.sin(2 * np.pi * 6.0 * t)  # AF8
    dummy_eeg[0, :] = 10.0 * np.sin(2 * np.pi * 10.0 * t) # TP9
    dummy_eeg[3, :] = 10.0 * np.sin(2 * np.pi * 10.0 * t) # TP10

    features = extractor.extract_features_from_window(dummy_eeg)

    # Check that key cognitive markers exist
    assert "frontal_theta_alpha_ratio" in features
    assert "frontal_engagement_index" in features
    assert "frontal_alpha_asymmetry" in features
    assert "AF7_theta_abs" in features
    assert "AF8_alpha_abs" in features
    assert "AF7_hjorth_activity" in features
    assert features["frontal_theta_alpha_ratio"] > 1.0


def test_dataset_loader_and_splits():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        loader = UniverseDatasetLoader(tmp_path)

        # Generate synthetic sessions for 3 participants
        for sub in ["UN_101", "UN_102", "UN_103"]:
            eeg_path, lbl_path = loader.generate_synthetic_universe_session(
                output_dir=tmp_path,
                participant_id=sub,
                session="Lab1",
                duration_seconds=20.0,
            )
            assert eeg_path.exists()
            assert lbl_path.exists()

        subs = loader.discover_participants()
        assert len(subs) == 3

        splits = loader.create_subject_independent_split(subs, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
        # Verify strict disjointness (no data leakage)
        train_set = set(splits.train_subjects)
        val_set = set(splits.val_subjects)
        test_set = set(splits.test_subjects)
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)


def test_eeg_service_integration():
    service = EEGService()

    # 4 channels x 4 seconds (1024 samples)
    n_samples = 256 * 4
    t = np.arange(n_samples) / 256.0
    sim_data = [
        (10.0 * np.sin(2 * np.pi * 10.0 * t)).tolist(),  # TP9
        (25.0 * np.sin(2 * np.pi * 6.0 * t)).tolist(),   # AF7 (high theta)
        (25.0 * np.sin(2 * np.pi * 6.0 * t)).tolist(),   # AF8 (high theta)
        (10.0 * np.sin(2 * np.pi * 10.0 * t)).tolist(),  # TP10
    ]

    req = PredictionRequest(
        session_id="test_session_001",
        eeg_data=sim_data,
        channel_names=["TP9", "AF7", "AF8", "TP10"],
        sampling_rate=256.0,
    )
    resp = service.process_request(req)
    assert resp.status == "success"
    assert len(resp.predictions) >= 1
    pred = resp.predictions[0]
    assert pred.workload in ["low", "medium", "high"]
    assert pred.theta_alpha_ratio is not None


if __name__ == "__main__":
    print("Running test_signal_quality_assessor()...")
    test_signal_quality_assessor()
    print("Running test_universe_eeg_preprocessor()...")
    test_universe_eeg_preprocessor()
    print("Running test_universe_feature_extractor()...")
    test_universe_feature_extractor()
    print("Running test_dataset_loader_and_splits()...")
    test_dataset_loader_and_splits()
    print("Running test_eeg_service_integration()...")
    test_eeg_service_integration()
    print("\nALL PREPROCESSING PIPELINE TESTS PASSED!")
