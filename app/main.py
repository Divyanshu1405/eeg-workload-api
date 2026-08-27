"""
Application entry point for the EEG Workload API.

This module:
- Creates the FastAPI application.
- Configures API metadata.
- Registers the application's API routes.

"""

from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(
    title="EEG Workload API",
    description="API for EEG signal processing and mental workload estimation.",
    version="0.1.0",
)

app.include_router(router)