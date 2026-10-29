"""
Request and response schemas for workload prediction.

This module defines the data structures exchanged between clients
(e.g., Next.js frontend) and the EEG Workload API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


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


class WorkloadPrediction(BaseModel):
    window: int = Field(default=1, description="Time window index")
    workload: str = Field(
        ...,
        description="Predicted mental workload category (e.g. 'low', 'medium', 'high', or 'placeholder_medium')",
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Prediction confidence score between 0.0 and 1.0 if supported by the model",
    )
    is_placeholder: bool = Field(
        default=True,
        description="True if this prediction is a placeholder mock rather than actual model inference",
    )


class PredictionResponse(BaseModel):
    session_id: str
    status: str = Field(
        default="success",
        description="Status of the prediction request",
    )
    model_source: str = Field(
        default="placeholder",
        description="Source of prediction: 'trained_model' if a trained model is loaded, otherwise 'placeholder'",
    )
    is_placeholder: bool = Field(
        default=True,
        description="Flag indicating whether the response is a placeholder or genuine model output",
    )
    message: Optional[str] = Field(
        default=None,
        description="Additional context regarding model availability or placeholder state",
    )
    feature_dim: Optional[int] = Field(
        default=None,
        description="Number of dimensions in the submitted feature vector, if provided",
    )
    predictions: List[WorkloadPrediction]