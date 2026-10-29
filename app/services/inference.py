"""
Inference service for EEG mental workload estimation.

This service manages loading approved production models from `models/production/`
and running predictions. If no model artifact is currently available in the
production directory, it returns clearly labeled placeholder responses.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.schemas.prediction import PredictionRequest, PredictionResponse, WorkloadPrediction


class InferenceService:
    def __init__(self, models_dir: Optional[Path] = None):
        if models_dir is None:
            # Default to repo_root/models/production
            repo_root = Path(__file__).resolve().parent.parent.parent
            self.models_dir = repo_root / "models" / "production"
        else:
            self.models_dir = Path(models_dir)

        self.model: Optional[Any] = None
        self.model_filename: Optional[str] = None
        self._load_model_if_available()

    def _load_model_if_available(self) -> None:
        """Scan models/production for an approved model artifact and load it if found."""
        if not self.models_dir.exists():
            return

        candidate_files = [
            f for f in self.models_dir.iterdir()
            if f.is_file() and not f.name.startswith(".") and f.name != ".gitkeep"
        ]

        if not candidate_files:
            return

        # Pick first available model file
        target_file = candidate_files[0]
        self.model_filename = target_file.name

        try:
            if target_file.suffix in [".pkl", ".joblib"]:
                try:
                    import joblib
                    self.model = joblib.load(target_file)
                except ImportError:
                    import pickle
                    with open(target_file, "rb") as f:
                        self.model = pickle.load(f)
            else:
                # Unsupported format or generic file
                self.model = None
        except Exception:
            self.model = None

    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about current model connectivity."""
        is_connected = self.model is not None
        return {
            "model_connected": is_connected,
            "models_directory": str(self.models_dir),
            "model_file": self.model_filename,
            "status": "ready" if is_connected else "no_model_found",
        }

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Run inference using the loaded model if available;
        otherwise return a clearly labelled placeholder response.
        """
        feature_dim = len(request.features) if request.features is not None else None

        # Scenario 1: Trained model is available and loaded
        if self.model is not None:
            try:
                features_input = [request.features] if request.features is not None else []
                raw_pred = self.model.predict(features_input)
                prediction_val = str(raw_pred[0]) if len(raw_pred) > 0 else "unknown"

                return PredictionResponse(
                    session_id=request.session_id,
                    status="success",
                    model_source="trained_model",
                    is_placeholder=False,
                    message=f"Prediction generated using production model: {self.model_filename}",
                    feature_dim=feature_dim,
                    predictions=[
                        WorkloadPrediction(
                            window=1,
                            workload=prediction_val,
                            confidence=None,
                            is_placeholder=False,
                        )
                    ],
                )
            except Exception as e:
                # Fallback if inference fails on invalid shape/data
                return PredictionResponse(
                    session_id=request.session_id,
                    status="error",
                    model_source="trained_model",
                    is_placeholder=False,
                    message=f"Error executing inference on loaded model: {e}",
                    feature_dim=feature_dim,
                    predictions=[],
                )

        # Scenario 2: No trained model available -> Return clearly labelled placeholder
        return PredictionResponse(
            session_id=request.session_id,
            status="success",
            model_source="placeholder",
            is_placeholder=True,
            message=(
                "PLACEHOLDER ONLY: No trained production model found in models/production/. "
                "Feature vector accepted, but EEG preprocessing and model training are pending."
            ),
            feature_dim=feature_dim,
            predictions=[
                WorkloadPrediction(
                    window=1,
                    workload="placeholder_medium",
                    confidence=0.5,
                    is_placeholder=True,
                )
            ],
        )


# Global service instance
inference_service = InferenceService()
