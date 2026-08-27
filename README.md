# EEG Workload API

Backend service for the EEG Mental Workload project.

This repository is intended to contain the EEG/ML pipeline, from data processing and model training to production inference through a FastAPI service.

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

The repository currently contains the initial backend/API skeleton.

Implemented:

- FastAPI application
- `GET /health` endpoint
- `POST /predict` endpoint with placeholder inference
- Pydantic request/response schemas
- Initial project structure and documentation

The `/predict` endpoint currently returns a placeholder prediction. Real EEG preprocessing, signal-quality assessment, feature extraction, model training, and model inference have not yet been implemented.

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
├── notebooks/                   # Exploratory and research notebooks
│
├── scripts/                     # Utility scripts
│
├── requirements.txt             # Python dependencies
├── .gitignore                   # Files excluded from Git
└── README.md                    # Project documentation
```

## Current Technology Stack

### Backend

- Python 3.11
- FastAPI
- Uvicorn
- Pydantic

## Planned Technology Stack

The following technologies are candidates for the EEG/ML pipeline and will be added as the corresponding parts of the project are implemented and evaluated.

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

### Development / Deployment

- Git
- GitHub
- Docker
- GitHub Actions

Specific ML/DL algorithms and deployment infrastructure are not considered finalized until they have been evaluated or implemented.

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

The `training/` directory is intended to contain code used to train and evaluate workload-estimation models.

The planned workflow is:

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

The project intends to prioritize subject-independent evaluation to investigate how well models generalize to participants whose EEG was not used during training.

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

## Development Principles

1. Build and validate the simplest working pipeline first.
2. Do not assume a particular ML/DL model will perform best.
3. Prevent data leakage during model evaluation.
4. Separate experimental code from production inference code.
5. Keep datasets and experimental artifacts out of Git.
6. Keep production preprocessing compatible with the production model.
7. Prefer reproducible experiments over undocumented manual changes.
8. Do not claim model performance before it has been experimentally evaluated.

## Deployment

The application is intended to be containerized using Docker.

The final hosting method will depend on available free infrastructure. Local or self-hosted execution will remain a fallback.

Deployment automation will be implemented after the production inference pipeline is established.

## Research Context

The project investigates EEG-based mental workload estimation, with particular focus on:

- Signal quality
- Subject-independent workload estimation
- Generalization to unseen participants
- User-specific calibration/personalization
- Comparison of classical machine learning and compact deep learning approaches

The project does not claim to measure an absolute percentage of brain capacity or provide medical diagnosis.
