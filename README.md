# EEG Workload API

Production FastAPI backend for wearable Muse S (`TP9`, `AF7`, `AF8`, `TP10` @ 256 Hz) EEG signal processing, signal quality assessment (SQA), per-user baseline calibration, and mental workload classification (`low` / `medium` / `high`).

- **Live API:** `https://eeg-workload-api.onrender.com`
- **Swagger UI:** `https://eeg-workload-api.onrender.com/docs`
- **Health Check:** `https://eeg-workload-api.onrender.com/health`

---

## Quickstart & Local Setup

Requires **Python 3.11+**:

```bash
# 1. Clone & navigate to repo
git clone https://github.com/Divyanshu1405/eeg-workload-api.git
cd eeg-workload-api

# 2. Create and activate virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies & run locally
pip install -r requirements.txt       # or requirements-prod.txt for minimal runtime
uvicorn app.main:app --reload --port 8000

# 4. Run test suite
python -m pytest -v
```

### Docker

```bash
docker build -t eeg-workload-api .
docker run -p 7860:7860 -e PORT=7860 eeg-workload-api
```

---

## API Endpoints

| Endpoint | Method | Format | Description |
| :--- | :---: | :--- | :--- |
| `/health` | `GET` | — | Service status and active production model metadata |
| `/predict` | `POST` | `application/json` | Workload prediction from raw EEG `[4 x N]` or feature vectors |
| `/predict/upload` | `POST` | `multipart/form-data` | Upload `.csv` or `.json` EEG file for end-to-end inference |
| `/calibrate` | `POST` | `application/json` | Store resting EEG baseline ($\ge$ 8s recommended) for a `user_id` |
| `/calibrate/upload` | `POST` | `multipart/form-data` | Upload resting `.csv` or `.json` to calibrate a `user_id` |
| `/calibrate/{user_id}` | `GET` / `DELETE` | — | Inspect or clear a stored resting baseline |

---

## Usage Examples

### 1. Predict from Raw EEG (JSON)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session-001",
    "user_id": "user-101",
    "sampling_rate": 256.0,
    "channel_names": ["TP9", "AF7", "AF8", "TP10"],
    "eeg_data": [
      [820.5, 822.1, 819.3, 821.0],
      [810.2, 812.0, 808.9, 811.4],
      [815.1, 814.8, 816.2, 815.5],
      [825.0, 824.2, 823.1, 824.8]
    ]
  }'
```

### 2. Predict from CSV Upload

Accepts standard Muse S CSV files (channels named `TP9, AF7, AF8, TP10` or `RAW_TP9, RAW_AF7, ...`):

```bash
curl -X POST "http://localhost:8000/predict/upload" \
  -F "file=@recording.csv" \
  -F "session_id=session-002" \
  -F "user_id=user-101"
```

### 3. Response Schema (Excerpt)

```json
{
  "session_id": "session-001",
  "calibrated": true,
  "signal_quality": {
    "is_usable": true,
    "overall_quality_score": 0.94,
    "bad_channels": []
  },
  "predictions": [
    {
      "window": 1,
      "start_time_sec": 0.0,
      "end_time_sec": 4.0,
      "workload": "medium",
      "confidence": 0.88,
      "theta_alpha_ratio": 1.42,
      "details": { "tar_delta_from_baseline": 0.15 }
    }
  ],
  "session_summary": {
    "duration_seconds": 8.0,
    "total_windows": 3,
    "dominant_workload": "medium",
    "level_percentages": { "low": 0.0, "medium": 66.7, "high": 33.3 }
  }
}
```

---

## Pipeline & Data Specs

- **Hardware & Channels:** Muse S 4-channel EEG (`TP9`, `AF7`, `AF8`, `TP10`) @ 256 Hz.
- **Windowing:** 4.0-second sliding windows (1,024 samples) with 50% overlap (2.0s step).
- **Preprocessing:** 0.5–45 Hz bandpass FIR filter, 50 Hz notch filter, Common Average Reference (CAR), and z-score normalization.
- **Signal Quality Assessment (SQA):** Checks SNR and amplitude clipping/flatline per window. Degraded windows are flagged `unusable` without producing misleading workload predictions.
- **Feature Extraction:** 112 features per window (band powers, Hjorth parameters, frontal alpha asymmetry, engagement index).
- **Personalized Calibration:** Baseline resting recordings establish subject-specific resting TAR and engagement index; subsequent inference computes normalized delta metrics.

---

## Repository Structure

```text
eeg-workload-api/
├── .github/workflows/
│   └── ci-deploy.yml          # Runs pytest, checks model loading, builds & smoke-tests Docker
├── app/
│   ├── api/routes.py          # FastAPI endpoints (/health, /predict, /calibrate)
│   ├── schemas/prediction.py  # Pydantic request/response schemas
│   ├── services/              # SQA, feature extraction, calibration, inference, file parsing
│   └── main.py                # FastAPI app & CORS middleware
├── models/production/         # Serialized model (.joblib) & feature metadata
├── scripts/                   # Preprocessing, evaluation, streaming test harness
├── tests/                     # Unit & integration test suite
├── training/                  # Core feature extraction, signal quality, and model definitions
├── Dockerfile                 # Slim Python 3.11 + XGBoost container
├── requirements-prod.txt      # Minimal production runtime dependencies
└── requirements.txt           # Local development & full training dependencies
```

---

## Model Benchmarks (UNIVERSE Dataset)

Evaluated strictly on 3 unseen test subjects (`UN_102`, `UN_109`, `UN_114`; 19,978 windows) after training on 17 subjects (112,515 windows):

| Model | Input | Test Accuracy | Test Macro F1 | Live `p50` Latency |
| :--- | :--- | :---: | :---: | :---: |
| **XGBoost (balanced, prod)** | 112 spectral/Hjorth features | `0.341` | `0.284` | **8.5 ms** |
| **Random Forest (balanced)** | 112 spectral/Hjorth features | `0.347` | `0.285` | — |
| **EEGNet (PyTorch CUDA)** | Raw `[4 x 1024]` waveforms | `0.278` | `0.251` | — |

To preprocess data or retrain models:
```bash
python scripts/preprocess_universe.py
python scripts/train_and_evaluate.py
```

---

## Deployment Notes (Render Free Tier)

- Automatically built and deployed from `main` using `Dockerfile` (`requirements-prod.txt`).
- Free instances sleep after 15 minutes of inactivity (~45–60s cold start on first request).
- `/calibrate` baselines are stored in memory and reset on container restart or redeploy.
