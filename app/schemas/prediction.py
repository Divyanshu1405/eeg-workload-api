"""
Request and response schemas for workload prediction.

This module defines the data structures exchanged between the
Next.js application and the EEG Workload API.
"""

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    session_id: str


class WorkloadPrediction(BaseModel):
    window: int
    workload: str


class PredictionResponse(BaseModel):
    session_id: str
    status: str
    predictions: list[WorkloadPrediction]