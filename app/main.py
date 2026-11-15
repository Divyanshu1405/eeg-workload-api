"""
Application entry point for the EEG Workload API.

This module:
- Creates the FastAPI application.
- Configures CORS middleware for frontend communication.
- Registers the application's API routes.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(
    title="EEG Workload API",
    description="API for EEG signal processing and mental workload estimation.",
    version="0.1.0",
)

# Enable CORS for Next.js frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
