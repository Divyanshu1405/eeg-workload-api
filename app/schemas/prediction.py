"""
Request and response schemas for workload prediction with UNIVERSE Muse S EEG and feature vector support.

This module defines the data structures exchanged between clients
(e.g., Next.js frontend) and the EEG Workload API.
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
    window: int = Field(default=1, description="Time window index")
    workload: str = Field(
        ...,
        description="Predicted mental workload category ('low', 'medium', 'high', 'unusable', or 'placeholder_medium')",
    )
    confidence: Optional[float] = Field(
        default=1.0,
        description="Prediction confidence score between 0.0 and 1.0",
    )
    quality_score: Optional[float] = Field(
        default=1.0,
        description="EEG signal quality score",
    )
    is_placeholder: bool = Field(
        default=True,
        description="True if this prediction is a placeholder mock rather than actual model inference",
    )
    theta_alpha_ratio: Optional[float] = None
    engagement_index: Optional[float] = None
    frontal_alpha_asymmetry: Optional[float] = None
    details: Optional[Dict[str, float]] = None


class PredictionRequest(BaseModel):
    session_id: str = Field(
        ...,
        description="Unique identifier for the recording session or subject",
        examples=["session-001"],
    )
    features: Optional[List[float]] = Field(
        default=None,
        description="Feature vector representing EEG characteristics (e.g. PSD band powers, ratios)",
        examples=[[0.24, 0.58, 1.12, 0.45, 0.15]],
    )
    # Optional raw EEG multi-channel array: [channels x samples]
    # Channels should be in order: TP9, AF7, AF8, TP10
    eeg_data: Optional[List[List[float]]] = Field(
        default=None,
        description="Raw EEG multi-channel array [channels x samples]",
    )
    channel_names: Optional[List[str]] = Field(
        default_factory=lambda: ["TP9", "AF7", "AF8", "TP10"],
        description="Channel names corresponding to eeg_data rows",
    )
    sampling_rate: float = Field(
        default=256.0,
        description="Sampling rate in Hz (default: 256.0 for Muse S)",
    )


class PredictionResponse(BaseModel):
    session_id: str
    status: str = Field(
        default="success",
        description="Status of the prediction request",
    )
    model_source: Optional[str] = Field(
        default="placeholder",
        description="Source of prediction: 'trained_model', 'eeg_pipeline', or 'placeholder'",
    )
    is_placeholder: Optional[bool] = Field(
        default=True,
        description="Flag indicating whether the response is a placeholder or genuine model output",
    )
    message: Optional[str] = Field(
        default=None,
        description="Additional context regarding model availability or processing details",
    )
    feature_dim: Optional[int] = Field(
        default=None,
        description="Number of dimensions in the submitted feature vector, if provided",
    )
    predictions: List[WorkloadPrediction]
    signal_quality: Optional[SignalQualitySchema] = None
