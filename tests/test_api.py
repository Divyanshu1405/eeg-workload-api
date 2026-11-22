import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

from app.main import app
from app.schemas.prediction import PredictionRequest
from app.services.calibration import calibration_service
from app.services.inference import InferenceService, inference_service


def _make_synthetic_eeg(duration_sec: float = 4.0, sr: float = 256.0, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    return [
        (820.0 + 15.0 * np.sin(2 * np.pi * 6.0 * t) + 5.0 * np.sin(2 * np.pi * 10.0 * t) + rng.normal(0, 2, n)).tolist()
        for _ in range(4)
    ]


class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("service", data)
        self.assertIsInstance(data["model_connected"], bool)
        self.assertIn(data["model_info"]["status"], ("ready", "no_model_found"))

    def test_predict_with_feature_vector_placeholder(self):
        with patch.object(inference_service, "model", None):
            resp = self.client.post(
                "/predict",
                json={"session_id": "session-test-001", "features": [0.12, 0.45, 0.89, -0.34, 1.2]},
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["model_source"], "placeholder")
        self.assertTrue(data["is_placeholder"])
        self.assertEqual(data["feature_dim"], 5)
        self.assertEqual(data["predictions"][0]["workload"], "placeholder_medium")

    def test_predict_with_raw_eeg_and_session_summary(self):
        resp = self.client.post(
            "/predict",
            json={
                "session_id": "session-raw-001",
                "sampling_rate": 256.0,
                "channel_names": ["TP9", "AF7", "AF8", "TP10"],
                "eeg_data": _make_synthetic_eeg(duration_sec=8.0),
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(data["signal_quality"]["is_usable"])
        self.assertIsNotNone(data["session_summary"])
        self.assertEqual(data["session_summary"]["duration_seconds"], 8.0)
        self.assertGreaterEqual(data["session_summary"]["total_windows"], 2)

    def test_predict_upload_csv(self):
        eeg = np.array(_make_synthetic_eeg(duration_sec=4.0)).T
        df = pd.DataFrame(eeg, columns=["TP9", "AF7", "AF8", "TP10"])
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        resp = self.client.post(
            "/predict/upload",
            data={"session_id": "upload-csv-01", "sampling_rate": "256.0"},
            files={"file": ("recording.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["session_id"], "upload-csv-01")
        self.assertEqual(data["status"], "success")
        self.assertEqual(len(data["predictions"]), 1)

    def test_calibration_lifecycle_and_calibrated_predict(self):
        uid = "test-user-cal-01"
        calibration_service.delete_baseline(uid)

        # 1. Calibrate via JSON
        cal_resp = self.client.post(
            "/calibrate",
            json={
                "user_id": uid,
                "sampling_rate": 256.0,
                "channel_names": ["TP9", "AF7", "AF8", "TP10"],
                "eeg_data": _make_synthetic_eeg(duration_sec=8.0, seed=10),
            },
        )
        self.assertEqual(cal_resp.status_code, 200)
        cal_data = cal_resp.json()
        self.assertTrue(cal_data["is_calibrated"])
        self.assertGreaterEqual(cal_data["usable_windows"], 1)

        # 2. Check GET /calibrate/{user_id}
        status_resp = self.client.get(f"/calibrate/{uid}")
        self.assertEqual(status_resp.status_code, 200)
        self.assertTrue(status_resp.json()["is_calibrated"])

        # 3. Predict with user_id -> calibrated=True + baseline deltas attached
        pred_resp = self.client.post(
            "/predict",
            json={
                "session_id": "cal-pred-01",
                "user_id": uid,
                "sampling_rate": 256.0,
                "channel_names": ["TP9", "AF7", "AF8", "TP10"],
                "eeg_data": _make_synthetic_eeg(duration_sec=4.0, seed=20),
            },
        )
        self.assertEqual(pred_resp.status_code, 200)
        pred_data = pred_resp.json()
        self.assertTrue(pred_data["calibrated"])
        self.assertIn("tar_delta_from_baseline", pred_data["predictions"][0]["details"])

        # 4. Delete calibration
        del_resp = self.client.delete(f"/calibrate/{uid}")
        self.assertEqual(del_resp.status_code, 200)
        self.assertFalse(del_resp.json()["is_calibrated"])

    def test_inference_service_with_trained_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_file = Path(tmp_dir) / "test_model.pkl"
            model_file.write_text("mock")

            service = InferenceService(models_dir=Path(tmp_dir))
            mock_model = MagicMock()
            mock_model.predict.return_value = ["high"]
            service.model = mock_model
            service.model_filename = "test_model.pkl"

            req = PredictionRequest(session_id="session-trained-001", features=[0.5, 0.8, 1.2])
            resp = service.predict(req)

            self.assertEqual(resp.status, "success")
            self.assertEqual(resp.model_source, "trained_model")
            self.assertFalse(resp.is_placeholder)
            self.assertEqual(resp.predictions[0].workload, "high")


if __name__ == "__main__":
    unittest.main()
