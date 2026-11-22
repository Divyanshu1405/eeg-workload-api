"""
Inference service for EEG mental workload estimation.
Loads approved production .joblib/.pkl models and their feature metadata from models/production/.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.schemas.prediction import PredictionRequest, PredictionResponse, WorkloadPrediction

logger = logging.getLogger("InferenceService")


class InferenceService:
    def __init__(self, models_dir: Optional[Path] = None):
        if models_dir is None:
            repo_root = Path(__file__).resolve().parent.parent.parent
            self.models_dir = repo_root / "models" / "production"
        else:
            self.models_dir = Path(models_dir)

        self.model: Optional[Any] = None
        self.model_filename: Optional[str] = None
        self.feature_names: Optional[List[str]] = None
        self.metadata: Dict[str, Any] = {}
        self._load_model_if_available()

    def _load_model_if_available(self) -> None:
        """Scan models/production for a serialized model artifact (.joblib/.pkl) and metadata."""
        if not self.models_dir.exists():
            return

        meta_path = self.models_dir / "model_metadata.json"
        preferred_type: Optional[str] = None
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                self.feature_names = self.metadata.get("features_used")
                preferred_type = self.metadata.get("model_type")
            except Exception as e:
                logger.warning("Failed to read model_metadata.json: %s", e)

        candidate_files = sorted(
            f for f in self.models_dir.iterdir()
            if f.is_file() and f.suffix in {".joblib", ".pkl"}
        )
        if not candidate_files:
            return

        target_file = candidate_files[0]
        if preferred_type:
            for f in candidate_files:
                if preferred_type in f.name:
                    target_file = f
                    break

        try:
            import joblib
            self.model = joblib.load(target_file)
            self.model_filename = target_file.name
            logger.info("Loaded production model: %s", self.model_filename)
        except Exception as e:
            logger.error("Failed to load model %s: %s", target_file.name, e)
            self.model = None
            self.model_filename = None

    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about current model connectivity."""
        is_connected = self.model is not None
        return {
            "model_connected": is_connected,
            "models_directory": str(self.models_dir),
            "model_file": self.model_filename,
            "feature_dim": len(self.feature_names) if self.feature_names else None,
            "status": "ready" if is_connected else "no_model_found",
        }

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Run inference on a pre-extracted feature vector."""
        feature_dim = len(request.features) if request.features is not None else None

        if self.model is not None:
            try:
                features_input = [request.features] if request.features is not None else []
                raw_pred = self.model.predict(features_input)
                prediction_val = str(raw_pred[0]) if len(raw_pred) > 0 else "unknown"

                confidence = None
                if hasattr(self.model, "predict_proba"):
                    try:
                        probs = self.model.predict_proba(features_input)
                        confidence = round(float(max(probs[0])), 3)
                    except Exception:
                        confidence = None

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
                            confidence=confidence,
                            is_placeholder=False,
                        )
                    ],
                )
            except Exception as e:
                return PredictionResponse(
                    session_id=request.session_id,
                    status="error",
                    model_source="trained_model",
                    is_placeholder=False,
                    message=f"Error executing inference on loaded model: {e}",
                    feature_dim=feature_dim,
                    predictions=[],
                )

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


inference_service = InferenceService()
