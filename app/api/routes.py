"""
HTTP API routes for the EEG Workload API.
Exposes health, JSON prediction, EEG file-upload prediction, and optional per-user calibration.
"""

from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.prediction import (
    CalibrationRequest,
    CalibrationResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.services.calibration import calibration_service
from app.services.eeg_service import eeg_service
from app.services.file_parser import parse_eeg_upload
from app.services.inference import inference_service

router = APIRouter()


@router.get("/health")
def health_check():
    """Health check endpoint exposing service health and model status."""
    model_info = inference_service.get_model_info()
    return {
        "status": "ok",
        "service": "EEG Workload API",
        "version": "0.2.0",
        "model_connected": model_info["model_connected"],
        "model_info": model_info,
    }


@router.post("/predict", response_model=PredictionResponse)
def predict_workload(request: PredictionRequest):
    """Mental workload prediction from JSON raw EEG arrays or pre-extracted feature vectors."""
    return eeg_service.process_request(request)


@router.post("/predict/upload", response_model=PredictionResponse)
async def predict_workload_upload(
    file: UploadFile = File(..., description="EEG recording file (.csv or .json)"),
    session_id: str = Form(default="uploaded-session"),
    user_id: Optional[str] = Form(default=None),
    sampling_rate: float = Form(default=256.0),
):
    """Upload a .csv or .json EEG recording and run full preprocessing + workload inference."""
    content = await file.read()
    try:
        eeg_data, channels, sr = parse_eeg_upload(content, file.filename or "", sampling_rate)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    req = PredictionRequest(
        session_id=session_id,
        user_id=user_id,
        eeg_data=eeg_data,
        channel_names=channels,
        sampling_rate=sr,
    )
    return eeg_service.process_request(req)


@router.post("/calibrate", response_model=CalibrationResponse)
def calibrate_user(request: CalibrationRequest):
    """Store a user's resting EEG baseline from a JSON raw EEG array."""
    return eeg_service.calibrate_user(request)


@router.post("/calibrate/upload", response_model=CalibrationResponse)
async def calibrate_user_upload(
    file: UploadFile = File(..., description="Resting baseline EEG file (.csv or .json)"),
    user_id: str = Form(..., description="Unique user identifier"),
    sampling_rate: float = Form(default=256.0),
):
    """Upload a resting baseline .csv or .json EEG file to calibrate a user."""
    content = await file.read()
    try:
        eeg_data, channels, sr = parse_eeg_upload(content, file.filename or "", sampling_rate)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    req = CalibrationRequest(
        user_id=user_id,
        eeg_data=eeg_data,
        channel_names=channels,
        sampling_rate=sr,
    )
    return eeg_service.calibrate_user(req)


@router.get("/calibrate/{user_id}", response_model=CalibrationResponse)
def get_calibration_status(user_id: str):
    """Check whether a user has an active resting EEG calibration baseline."""
    rec = calibration_service.get_baseline(user_id)
    if not rec:
        return CalibrationResponse(
            user_id=user_id,
            status="not_found",
            is_calibrated=False,
            message=f"No calibration baseline stored for {user_id}.",
        )
    return CalibrationResponse(
        user_id=user_id,
        status="success",
        is_calibrated=True,
        usable_windows=rec["usable_windows"],
        rejected_windows=rec["rejected_windows"],
        avg_quality_score=rec["avg_quality_score"],
        baseline_tar=rec["baseline_tar"],
        baseline_engagement=rec["baseline_engagement"],
        created_at=rec["created_at"],
        message="Active calibration baseline found.",
    )


@router.delete("/calibrate/{user_id}", response_model=CalibrationResponse)
def clear_calibration(user_id: str):
    """Delete a user's stored resting EEG baseline."""
    deleted = calibration_service.delete_baseline(user_id)
    return CalibrationResponse(
        user_id=user_id,
        status="deleted" if deleted else "not_found",
        is_calibrated=False,
        message="Calibration cleared." if deleted else f"No calibration found for {user_id}.",
    )
