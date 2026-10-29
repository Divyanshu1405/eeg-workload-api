
"""
HTTP API routes for the EEG Workload API.

This module:
- Defines the endpoints exposed to clients.
- Receives and validates API requests.
- Dispatches inference requests to the InferenceService.
"""

from fastapi import APIRouter

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.inference import inference_service

router = APIRouter()


@router.get("/health")
def health_check():
    """Health check endpoint exposing service health and model status."""
    model_info = inference_service.get_model_info()
    return {
        "status": "ok",
        "service": "EEG Workload API",
        "version": "0.1.0",
        "model_connected": model_info["model_connected"],
        "model_info": model_info,
    }


@router.post("/predict", response_model=PredictionResponse)
def predict_workload(request: PredictionRequest):
    """
    Mental workload prediction endpoint.
    """
    return inference_service.predict(request)
