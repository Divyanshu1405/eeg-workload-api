"""
HTTP API routes for the EEG Workload API.

This module:
- Defines the endpoints exposed to clients.
- Receives and validates API requests.
- Passes actual processing work to service modules.

"""

from fastapi import APIRouter

from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    WorkloadPrediction,
)


router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/predict", response_model=PredictionResponse)
def predict_workload(request: PredictionRequest):
    return PredictionResponse(
        session_id=request.session_id,
        status="success",
        predictions=[
            WorkloadPrediction(
                window=1,
                workload="medium",
            )
        ],
    )