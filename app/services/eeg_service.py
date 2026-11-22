"""
Production EEG Processing, Calibration, and Inference Service.
Handles signal quality assessment, preprocessing, feature extraction, optional
per-user calibration, and workload estimation via InferenceService.
"""

from typing import List, Optional
import numpy as np

from app.schemas.prediction import (
    CalibrationRequest,
    CalibrationResponse,
    PredictionRequest,
    PredictionResponse,
    SessionSummarySchema,
    SignalQualitySchema,
    WorkloadPrediction,
)
from app.services.calibration import calibration_service
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

    def _segment_eeg(self, eeg_arr: np.ndarray, channels: List[str], sampling_rate: float):
        window_samples = int(sampling_rate * 4.0)
        if eeg_arr.shape[1] > window_samples:
            return self.preprocessor.segment_and_preprocess_continuous(
                eeg_arr, channel_names=channels
            )
        return [
            self.preprocessor.preprocess_window(
                eeg_arr, channel_names=channels, window_index=0
            )
        ]

    def calibrate_user(self, request: CalibrationRequest) -> CalibrationResponse:
        """Process a user's resting EEG recording and store their feature baseline."""
        eeg_arr = np.array(request.eeg_data, dtype=np.float64)
        channels = request.channel_names or ["TP9", "AF7", "AF8", "TP10"]

        if eeg_arr.ndim != 2 or eeg_arr.shape[0] != len(channels):
            return CalibrationResponse(
                user_id=request.user_id,
                status="error",
                is_calibrated=False,
                message="Invalid eeg_data shape: expected 2D [channels x samples].",
            )

        if eeg_arr.shape[1] < int(request.sampling_rate * 1.0):
            return CalibrationResponse(
                user_id=request.user_id,
                status="error",
                is_calibrated=False,
                message="Calibration recording too short (minimum 1 second, recommended >= 8 seconds).",
            )

        windows = self._segment_eeg(eeg_arr, channels, request.sampling_rate)
        usable_feats = []
        rejected = 0
        qual_scores = []

        for win in windows:
            if not win.is_usable:
                rejected += 1
                continue
            feats = self.feature_extractor.extract_features_from_window(win.data, win.channel_names)
            usable_feats.append(feats)
            qual_scores.append(win.quality_metrics.overall_quality_score)

        if not usable_feats:
            return CalibrationResponse(
                user_id=request.user_id,
                status="error",
                is_calibrated=False,
                usable_windows=0,
                rejected_windows=rejected,
                message="No usable EEG windows passed signal quality assessment. Please check headband fit and retry.",
            )

        record = calibration_service.save_baseline(
            user_id=request.user_id,
            feature_dicts=usable_feats,
            rejected_windows=rejected,
            avg_quality_score=float(np.mean(qual_scores)),
        )

        return CalibrationResponse(
            user_id=request.user_id,
            status="success",
            is_calibrated=True,
            usable_windows=record["usable_windows"],
            rejected_windows=record["rejected_windows"],
            avg_quality_score=record["avg_quality_score"],
            baseline_tar=record["baseline_tar"],
            baseline_engagement=record["baseline_engagement"],
            created_at=record["created_at"],
            message=f"Stored resting baseline for {request.user_id} from {record['usable_windows']} usable window(s).",
        )

    def estimate_workload(
        self,
        features: dict,
        user_id: Optional[str] = None,
    ) -> tuple[str, float, Optional[dict]]:
        """
        Estimates workload using the loaded production model if available,
        otherwise falls back to cognitive marker heuristics.
        """
        model_input_feats = features
        details = None

        # Apply baseline normalization to model inputs only if the active model expects it;
        # always attach relative baseline deltas for frontend visibility when calibrated.
        if user_id and calibration_service.get_baseline(user_id) is not None:
            baseline = calibration_service.get_baseline(user_id)
            if inference_service.metadata.get("requires_calibration", False):
                model_input_feats = calibration_service.normalize_features(user_id, features)
            tar_now = features.get("frontal_theta_alpha_ratio", 0.0)
            eng_now = features.get("frontal_engagement_index", 0.0)
            details = {
                "tar_delta_from_baseline": round(tar_now - baseline["baseline_tar"], 3),
                "engagement_delta_from_baseline": round(eng_now - baseline["baseline_engagement"], 3),
            }

        feature_keys = inference_service.feature_names or sorted(model_input_feats.keys())
        feature_vector = [float(model_input_feats.get(k, 0.0)) for k in feature_keys]

        if inference_service.model is not None:
            try:
                raw_pred = inference_service.model.predict([feature_vector])
                predicted_level = str(raw_pred[0])
                conf = 0.85
                if hasattr(inference_service.model, "predict_proba"):
                    probs = inference_service.model.predict_proba([feature_vector])
                    conf = float(max(probs[0]))
                return predicted_level, conf, details
            except Exception:
                pass

        # Heuristic fallback
        tar = features.get("frontal_theta_alpha_ratio", 1.0)
        eng = features.get("frontal_engagement_index", 0.5)
        score = 0.65 * (tar / 1.5) + 0.35 * (eng / 0.8)

        if tar >= 1.35 or score >= 1.2:
            return "high", min(0.95, 0.60 + 0.35 * min(1.0, tar / 2.0)), details
        elif tar <= 0.85 or score <= 0.65:
            return "low", min(0.95, 0.60 + 0.35 * max(0.0, 1.0 - tar)), details
        else:
            return "medium", 0.75, details

    def process_request(self, request: PredictionRequest) -> PredictionResponse:
        """Processes an incoming API request (raw EEG or pre-extracted features)."""
        if request.features is not None and (request.eeg_data is None or len(request.eeg_data) == 0):
            return inference_service.predict(request)

        if request.eeg_data is None or len(request.eeg_data) == 0:
            return PredictionResponse(
                session_id=request.session_id,
                user_id=request.user_id,
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
                user_id=request.user_id,
                status="error: invalid array dimension (must be 2D [channels, samples])",
                predictions=[],
            )

        channels = request.channel_names or ["TP9", "AF7", "AF8", "TP10"]
        if eeg_arr.shape[0] != len(channels):
            return PredictionResponse(
                session_id=request.session_id,
                user_id=request.user_id,
                status=f"error: channel mismatch ({eeg_arr.shape[0]} rows vs {len(channels)} channel names)",
                predictions=[],
            )

        min_samples = int(request.sampling_rate * 1.0)
        if eeg_arr.shape[1] < min_samples:
            return PredictionResponse(
                session_id=request.session_id,
                user_id=request.user_id,
                status=f"error: recording too short ({eeg_arr.shape[1]} samples)",
                predictions=[],
            )

        windows = self._segment_eeg(eeg_arr, channels, request.sampling_rate)
        predictions: List[WorkloadPrediction] = []
        usable_count = 0
        rejected_count = 0
        avg_quality = 0.0
        level_counts = {"low": 0, "medium": 0, "high": 0, "unusable": 0}

        model_connected = inference_service.model is not None
        is_calibrated = bool(request.user_id and calibration_service.get_baseline(request.user_id) is not None)

        for win in windows:
            avg_quality += win.quality_metrics.overall_quality_score
            if not win.is_usable:
                rejected_count += 1
                level_counts["unusable"] += 1
                predictions.append(
                    WorkloadPrediction(
                        window=win.window_index + 1,
                        start_time_sec=round(win.start_time_sec, 2),
                        end_time_sec=round(win.end_time_sec, 2),
                        workload="unusable",
                        confidence=0.0,
                        quality_score=round(win.quality_metrics.overall_quality_score, 3),
                        is_placeholder=not model_connected,
                    )
                )
                continue

            usable_count += 1
            feats = self.feature_extractor.extract_features_from_window(
                win.data, win.channel_names
            )
            wl_level, conf, details = self.estimate_workload(feats, user_id=request.user_id)
            level_counts[wl_level] = level_counts.get(wl_level, 0) + 1

            tar = feats.get("frontal_theta_alpha_ratio")
            eng = feats.get("frontal_engagement_index")
            faa = feats.get("frontal_alpha_asymmetry")

            predictions.append(
                WorkloadPrediction(
                    window=win.window_index + 1,
                    start_time_sec=round(win.start_time_sec, 2),
                    end_time_sec=round(win.end_time_sec, 2),
                    workload=wl_level,
                    confidence=round(conf, 3),
                    quality_score=round(win.quality_metrics.overall_quality_score, 3),
                    is_placeholder=not model_connected,
                    theta_alpha_ratio=round(tar, 3) if tar is not None else None,
                    engagement_index=round(eng, 3) if eng is not None else None,
                    frontal_alpha_asymmetry=round(faa, 3) if faa is not None else None,
                    details=details,
                )
            )

        total_wins = max(1, len(windows))
        mean_qual = round(avg_quality / total_wins, 3)
        first_metrics = windows[0].quality_metrics
        sq_schema = SignalQualitySchema(
            is_usable=usable_count > 0,
            overall_quality_score=mean_qual,
            bad_channels=first_metrics.bad_channel_names,
            rejection_reasons=first_metrics.rejection_reasons,
        )

        usable_level_counts = {k: level_counts[k] for k in ("low", "medium", "high")}
        dominant = (
            max(usable_level_counts, key=usable_level_counts.get)
            if usable_count > 0
            else "unusable"
        )
        summary = SessionSummarySchema(
            duration_seconds=round(eeg_arr.shape[1] / request.sampling_rate, 2),
            total_windows=len(windows),
            usable_windows=usable_count,
            rejected_windows=rejected_count,
            dominant_workload=dominant,
            level_percentages={k: round((v / total_wins) * 100.0, 1) for k, v in level_counts.items()},
            avg_quality_score=mean_qual,
        )

        return PredictionResponse(
            session_id=request.session_id,
            user_id=request.user_id,
            calibrated=is_calibrated,
            status="success",
            model_source="trained_model" if model_connected else "eeg_pipeline_heuristic",
            is_placeholder=not model_connected,
            message=(
                f"Processed {len(windows)} window(s) ({usable_count} usable) using trained model: {inference_service.model_filename}"
                if model_connected
                else f"Processed {len(windows)} window(s) ({usable_count} usable) using pipeline heuristics."
            ),
            predictions=predictions,
            signal_quality=sq_schema,
            session_summary=summary,
        )


eeg_service = EEGService()
