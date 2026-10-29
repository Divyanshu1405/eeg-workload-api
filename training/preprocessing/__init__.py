"""
Preprocessing and feature extraction pipeline for the UNIVERSE EEG dataset.
UNIVERSE: UNobtrusIVE measuRement of mental workload and stress in uncontrolled environments.
"""

from training.preprocessing.signal_quality import (
    SignalQualityAssessor,
    SignalQualityMetrics,
)
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

__all__ = [
    "SignalQualityAssessor",
    "SignalQualityMetrics",
    "UniverseEEGPreprocessor",
    "PreprocessingConfig",
    "UniverseFeatureExtractor",
    "FeatureConfig",
    "UniverseDatasetLoader",
    "DatasetSplit",
]
