"""
Request and response schemas for the EEG Workload API (Next.js frontend contract).
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ChannelMetricsSchema(BaseModel):
    channel_name: str
    is_valid: bool
    snr_db: float
    status: str
    reason: Optional[str] = None


class SignalQualitySchema(BaseModel):
    is_usable: bool
    overall_quality_score: float
    bad_channels: List[str] = Field(default_factory=list)
    rejection_reasons: List[str] = Field(default_factory=list)


class WorkloadPrediction(BaseModel):
    window: int = Field(default=1, description="1-based time window index")
    start_time_sec: Optional[float] = Field(default=None, description="Window start time in seconds")
    end_time_sec: Optional[float] = Field(default=None, description="Window end time in seconds")
    workload: str = Field(
        ...,
        description="Predicted mental workload ('low', 'medium', 'high', 'unusable', or 'placeholder_medium')",
    )
    confidence: Optional[float] = Field(default=1.0, description="Prediction confidence (0.0 to 1.0)")
    quality_score: Optional[float] = Field(default=1.0, description="Window EEG signal quality score")
    is_placeholder: bool = Field(default=True)
    theta_alpha_ratio: Optional[float] = None
    engagement_index: Optional[float] = None
    frontal_alpha_asymmetry: Optional[float] = None
    details: Optional[Dict[str, float]] = None


class SessionSummarySchema(BaseModel):
    duration_seconds: float
    total_windows: int
    usable_windows: int
    rejected_windows: int
    dominant_workload: str
    level_percentages: Dict[str, float]
    avg_quality_score: float


class PredictionRequest(BaseModel):
    session_id: str = Field(..., examples=["session-001"])
    user_id: Optional[str] = Field(
        default=None,
        description="Optional user ID to apply stored resting baseline calibration",
        examples=["user-101"],
    )
    features: Optional[List[float]] = Field(default=None)
    eeg_data: Optional[List[List[float]]] = Field(
        default=None,
        description="Raw EEG multi-channel array [channels x samples]",
    )
    channel_names: Optional[List[str]] = Field(
        default_factory=lambda: ["TP9", "AF7", "AF8", "TP10"],
    )
    sampling_rate: float = Field(default=256.0)


class PredictionResponse(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    calibrated: bool = Field(
        default=False,
        description="True if a stored resting baseline for user_id was found and applied",
    )
    status: str = Field(default="success")
    model_source: Optional[str] = Field(default="placeholder")
    is_placeholder: Optional[bool] = Field(default=True)
    message: Optional[str] = None
    feature_dim: Optional[int] = None
    predictions: List[WorkloadPrediction]
    signal_quality: Optional[SignalQualitySchema] = None
    session_summary: Optional[SessionSummarySchema] = None


class CalibrationRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier", examples=["user-101"])
    eeg_data: List[List[float]] = Field(
        ...,
        description="Resting/baseline raw EEG array [channels x samples], ideally >= 8 seconds",
    )
    channel_names: Optional[List[str]] = Field(
        default_factory=lambda: ["TP9", "AF7", "AF8", "TP10"],
    )
    sampling_rate: float = Field(default=256.0)


class CalibrationResponse(BaseModel):
    user_id: str
    status: str
    is_calibrated: bool
    usable_windows: int = 0
    rejected_windows: int = 0
    avg_quality_score: Optional[float] = None
    baseline_tar: Optional[float] = None
    baseline_engagement: Optional[float] = None
    created_at: Optional[str] = None
    message: Optional[str] = None
