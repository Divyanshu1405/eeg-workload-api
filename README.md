# EEG Workload API

Backend service for the EEG Mental Workload project.

This repository contains the complete EEG/ML pipeline, from data processing and model training to production inference through a FastAPI service.

## Project Scope

The system is intended to:

- Process recorded EEG data
- Assess EEG signal quality
- Preprocess EEG signals
- Extract relevant features
- Train and evaluate workload-estimation models
- Perform workload inference using an approved production model
- Provide workload predictions through a REST API
- Support future user-specific calibration/personalization

The project uses publicly available EEG datasets. Dataset files, experiment outputs, and development model artifacts are kept locally and are not committed to Git.

## Current Status

The API skeleton is currently implemented.

Available endpoints:

- `GET /health` — checks whether the API is running.
- `POST /predict` — currently returns a placeholder workload prediction.

Real EEG preprocessing and model inference will be added as the ML pipeline is developed.

## Repository Structure

```text
eeg-workload-api/
│
├── app/                         # Production API and inference code
│   ├── api/                     # HTTP API routes
│   ├── schemas/                 # API request/response schemas
│   ├── services/                # EEG processing and inference services
│   └── main.py                  # FastAPI application entry point
│
├── training/                    # Model training and evaluation pipeline
│
├── data/
│   ├── raw/                     # Raw datasets - not committed
│   └── processed/               # Processed datasets - not committed
│
├── experiments/                 # Local experiment outputs - not committed
│
├── models/
│   └── production/              # Approved production model
│
├── notebooks/                   # Research and exploratory analysis
│
├── tests/                       # Automated tests
│
├── scripts/                     # Utility scripts
│
├── .github/
│   └── workflows/               # GitHub Actions
│
├── Dockerfile                   # Container configuration
├── requirements.txt             # Python dependencies
├── .gitignore                   # Files excluded from Git
└── README.md                    # Project documentation
```

## Technology Stack

### Backend

- Python 3.11
- FastAPI
- Uvicorn
- Pydantic

### EEG / Scientific Computing

- MNE-Python
- NumPy
- SciPy
- pandas

### Machine Learning

- scikit-learn
- XGBoost
- Random Forest

### Deep Learning

- PyTorch
- EEGNet

### Development

- Git
- GitHub
- Docker
- GitHub Actions

## Local Setup

### Requirements

- Python 3.11
- Git

### 1. Clone the repository

```powershell
git clone <repository-url>
cd eeg-workload-api
```

### 2. Create the virtual environment

```powershell
py -3.11 -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 5. Run the API

```powershell
uvicorn app.main:app --reload
```

The API will be available at:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

## API

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

### Workload Prediction

```http
POST /predict
```

Current request:

```json
{
  "session_id": "demo-001"
}
```

Current response:

```json
{
  "session_id": "demo-001",
  "status": "success",
  "predictions": [
    {
      "window": 1,
      "workload": "medium"
    }
  ]
}
```

The prediction is currently a placeholder and does not perform real EEG inference.

## Planned EEG Pipeline

```text
EEG Recording
      ↓
Data Validation
      ↓
Signal Quality Assessment
      ↓
EEG Preprocessing
      ↓
Windowing
      ↓
Feature Extraction
      ↓
Workload Model
      ↓
Workload Prediction
      ↓
API Response
      ↓
Next.js Application
```

## Training Pipeline

The `training/` directory contains code used to train and evaluate workload-estimation models.

The general workflow will be:

```text
Public Dataset
      ↓
Data Loading
      ↓
Preprocessing
      ↓
Feature Extraction
      ↓
Model Training
      ↓
Subject-Independent Evaluation
      ↓
Model Selection
      ↓
Production Model
```

The project will prioritize subject-independent evaluation to investigate how well models generalize to participants whose EEG was not used during training.

## Dataset Policy

Datasets are not stored in Git.

Place downloaded datasets under:

```text
data/raw/
```

Processed data should be placed under:

```text
data/processed/
```

Both locations are ignored by Git.

The repository should contain only the folder structure and documentation required to reproduce the setup.

## Model Policy

Experimental models are not committed to Git.

During development, trained model artifacts remain local.

The approved production model will be placed under:

```text
models/production/
```

and explicitly committed to Git only when it is selected for deployment.

The production model must remain compatible with the preprocessing and feature-extraction pipeline used during training.

Model metadata will be maintained alongside the production model to record information such as:

- Model version
- Dataset
- Feature version
- Preprocessing version
- Workload classes

## Production Code Ownership

The following components are considered production-critical:

- EEG preprocessing
- Signal-quality assessment
- Feature extraction
- Model loading
- Model inference
- API schemas
- API contract

Changes to production-critical components should be reviewed before being merged.

Training and experimental code may evolve more freely, provided it does not unintentionally modify the production pipeline.

## API Contract

The Next.js application communicates with this service through HTTP/JSON.

Changes to API request or response schemas must be coordinated with the frontend repository.

The API should remain independent of the Next.js application and should not contain frontend-specific logic.

## Deployment

The application is intended to be containerized using Docker.

The final hosting method will depend on available free infrastructure. Local or self-hosted execution will remain a fallback.

Production deployment automation will be handled through GitHub Actions.

A production deployment is expected to be triggered when an approved production model is pushed to the `main` branch.

## Research Context

The project investigates EEG-based mental workload estimation, with particular focus on:

- Signal quality
- Subject-independent workload estimation
- Generalization to unseen participants
- User-specific calibration/personalization
- Comparison of classical machine learning and compact deep learning approaches

The project does not claim to measure an absolute percentage of brain capacity or provide medical diagnosis.
