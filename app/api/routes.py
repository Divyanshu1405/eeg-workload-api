"""
HTTP API routes for the EEG Workload API.

This module:
- Defines the endpoints exposed to clients.
- Receives and validates API requests.
- Passes actual processing work to service modules.

"""

from fastapi import APIRouter


router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}