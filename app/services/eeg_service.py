"""
Production EEG Processing and Inference Service for the UNIVERSE Dataset.
Handles signal quality assessment, preprocessing, feature extraction, and workload estimation,
integrating with the trained production model via InferenceService.
"""

from typing import List
import numpy as np

from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    SignalQualitySchema,
    WorkloadPrediction,
)
from app.services.inference import inference_service
from training.preprocessing.universe_preprocessor import (
    UniverseEEGPreprocessor,
    PreprocessingConfig,
)
from training.preprocessing.feature_extraction import (
    UniverseFeatureExtractor,
    FeatureConfig,
)


class EEGService:
    """High-performance production service for processing wearable Muse S EEG data."""

    def __init__(self):
        self.preprocessor = UniverseEEGPreprocessor(
            PreprocessingConfig(
                sampling_rate=256.0,
                channels=["TP9", "AF7", "AF8", "TP10"],
                bandpass_low=0.5,
                bandpass_high=45.0,
                notch_freq=50.0,
                apply_car=True,
                normalization="zscore",
            )
        )
        self.feature_extractor = UniverseFeatureExtractor(
            FeatureConfig(
                sampling_rate=256.0,
                channel_names=["TP9", "AF7", "AF8", "TP10"],
            )
        )

    def estimate_workload(self, features: dict) -> tuple[str, float]:
        """
        Estimates workload using the loaded production model if available,
        otherwise falls back to validated cognitive marker heuristics.
        """
        feature_keys = inference_service.feature_names or sorted(features.keys())
        feature_vector = [float(features.get(k, 0.0)) for k in feature_keys]

        if inference_service.model is not None:
            try:
                raw_pred = inference_service.model.predict([feature_vector])
                predicted_level = str(raw_pred[0])
                conf = 0.85
                if hasattr(inference_service.model, "predict_proba"):
                    probs = inference_service.model.predict_proba([feature_vector])
                    conf = float(max(probs[0]))
                return predicted_level, conf
            except Exception:
                pass

        # Heuristic fallback
        tar = features.get("frontal_theta_alpha_ratio", 1.0)
        eng = features.get("frontal_engagement_index", 0.5)
        score = 0.65 * (tar / 1.5) + 0.35 * (eng / 0.8)

        if tar >= 1.35 or score >= 1.2:
            return "high", min(0.95, 0.60 + 0.35 * min(1.0, tar / 2.0))
        elif tar <= 0.85 or score <= 0.65:
            return "low", min(0.95, 0.60 + 0.35 * max(0.0, 1.0 - tar))
        else:
            return "medium", 0.75

    def process_request(self, request: PredictionRequest) -> PredictionResponse:
        """Processes an incoming API request (raw EEG or pre-extracted features)."""
        if request.features is not None and (request.eeg_data is None or len(request.eeg_data) == 0):
            return inference_service.predict(request)

        if request.eeg_data is None or len(request.eeg_data) == 0:
            return PredictionResponse(
                session_id=request.session_id,
                status="success",
                model_source="placeholder",
                is_placeholder=True,
                predictions=[
                    WorkloadPrediction(
                        window=1,
                        workload="medium",
                        confidence=0.75,
                        quality_score=1.0,
                        is_placeholder=True,
                    )
                ],
            )

        eeg_arr = np.array(request.eeg_data, dtype=np.float64)
        if eeg_arr.ndim != 2:
            return PredictionResponse(
                session_id=request.session_id,
                status="error: invalid array dimension (must be 2D [channels, samples])",
                predictions=[],
            )

        channels = request.channel_names or ["TP9", "AF7", "AF8", "TP10"]
        if eeg_arr.shape[0] != len(channels):
            return PredictionResponse(
                session_id=request.session_id,
                status=f"error: channel mismatch ({eeg_arr.shape[0]} rows vs {len(channels)} channel names)",
                predictions=[],
            )

        min_samples = int(request.sampling_rate * 1.0)
        if eeg_arr.shape[1] < min_samples:
            return PredictionResponse(
                session_id=request.session_id,
                status=f"error: recording too short ({eeg_arr.shape[1]} samples)",
                predictions=[],
            )

        window_samples = int(request.sampling_rate * 4.0)
        if eeg_arr.shape[1] > window_samples:
            windows = self.preprocessor.segment_and_preprocess_continuous(
                eeg_arr, channel_names=channels
            )
        else:
            windows = [
                self.preprocessor.preprocess_window(
                    eeg_arr, channel_names=channels, window_index=0
                )
            ]

        predictions: List[WorkloadPrediction] = []
        overall_usable = True
        avg_quality = 0.0
        model_connected = inference_service.model is not None

        for win in windows:
            if not win.is_usable:
                overall_usable = False
                predictions.append(
                    WorkloadPrediction(
                        window=win.window_index + 1,
                        workload="unusable",
                        confidence=0.0,
                        quality_score=win.quality_metrics.overall_quality_score,
                        is_placeholder=not model_connected,
                    )
                )
                continue

            feats = self.feature_extractor.extract_features_from_window(
                win.data, win.channel_names
            )
            wl_level, conf = self.estimate_workload(feats)

            tar = feats.get("frontal_theta_alpha_ratio")
            eng = feats.get("frontal_engagement_index")
            faa = feats.get("frontal_alpha_asymmetry")
            avg_quality += win.quality_metrics.overall_quality_score

            predictions.append(
                WorkloadPrediction(
                    window=win.window_index + 1,
                    workload=wl_level,
                    confidence=round(conf, 3),
                    quality_score=round(win.quality_metrics.overall_quality_score, 3),
                    is_placeholder=not model_connected,
                    theta_alpha_ratio=round(tar, 3) if tar is not None else None,
                    engagement_index=round(eng, 3) if eng is not None else None,
                    frontal_alpha_asymmetry=round(faa, 3) if faa is not None else None,
                )
            )

        first_metrics = windows[0].quality_metrics
        sq_schema = SignalQualitySchema(
            is_usable=overall_usable,
            overall_quality_score=round(avg_quality / max(1, len(windows)), 3),
            bad_channels=first_metrics.bad_channel_names,
            rejection_reasons=first_metrics.rejection_reasons,
        )

        return PredictionResponse(
            session_id=request.session_id,
            status="success",
            model_source="trained_model" if model_connected else "eeg_pipeline_heuristic",
            is_placeholder=not model_connected,
            message=(
                f"Successfully processed {len(windows)} windows using trained model: {inference_service.model_filename}"
                if model_connected
                else f"Successfully processed {len(windows)} windows using pipeline heuristics."
            ),
            predictions=predictions,
            signal_quality=sq_schema,
        )


eeg_service = EEGService()
