"""
Request and response schemas for workload prediction with UNIVERSE Muse S EEG support.
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
    window: int
    workload: str  # 'low', 'medium', 'high', 'unusable'
    confidence: float = 1.0
    quality_score: float = 1.0
    theta_alpha_ratio: Optional[float] = None
    engagement_index: Optional[float] = None
    frontal_alpha_asymmetry: Optional[float] = None
    details: Optional[Dict[str, float]] = None


class PredictionRequest(BaseModel):
    session_id: str
    # Optional raw EEG multi-channel array: [channels x samples]
    # Channels should be in order: TP9, AF7, AF8, TP10
    eeg_data: Optional[List[List[float]]] = None
    channel_names: Optional[List[str]] = Field(
        default_factory=lambda: ["TP9", "AF7", "AF8", "TP10"]
    )
    sampling_rate: float = 256.0


class PredictionResponse(BaseModel):
    session_id: str
    status: str
    predictions: List[WorkloadPrediction]
    signal_quality: Optional[SignalQualitySchema] = None