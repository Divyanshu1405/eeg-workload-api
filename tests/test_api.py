import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from app.main import app
from app.schemas.prediction import PredictionRequest
from app.services.inference import InferenceService


class TestAPIEndpoints(unittest.IsolatedAsyncioTestCase):
    async def _call_endpoint(self, path: str, method: str = "GET", body: bytes = b""):
        response_body = []
        status_code = None

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": [
                (b"host", b"testserver"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
            "scheme": "http",
        }

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            elif message["type"] == "http.response.body":
                response_body.append(message.get("body", b""))

        await app(scope, receive, send)
        data = json.loads(b"".join(response_body).decode("utf-8"))
        return status_code, data

    async def test_health_check(self):
        status, data = await self._call_endpoint("/health", "GET")
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "ok")
        self.assertIn("service", data)
        self.assertIn("model_connected", data)
        self.assertFalse(data["model_connected"])

    async def test_predict_with_feature_vector_placeholder(self):
        payload = json.dumps({
            "session_id": "session-test-001",
            "features": [0.12, 0.45, 0.89, -0.34, 1.2]
        }).encode("utf-8")
        status, data = await self._call_endpoint("/predict", "POST", payload)
        self.assertEqual(status, 200)
        self.assertEqual(data["session_id"], "session-test-001")
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["model_source"], "placeholder")
        self.assertTrue(data["is_placeholder"])
        self.assertEqual(data["feature_dim"], 5)
        self.assertIn("PLACEHOLDER ONLY", data["message"])
        self.assertIsInstance(data["predictions"], list)
        self.assertGreater(len(data["predictions"]), 0)
        self.assertTrue(data["predictions"][0]["is_placeholder"])
        self.assertEqual(data["predictions"][0]["workload"], "placeholder_medium")

    def test_inference_service_with_trained_model(self):
        """Test the trained model branch when a model is present."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_file = Path(tmp_dir) / "test_model.pkl"
            # Create a mock model file
            model_file.write_text("mock")

            service = InferenceService(models_dir=Path(tmp_dir))
            # Inject a mock model object with a predict method
            mock_model = MagicMock()
            mock_model.predict.return_value = ["high"]
            service.model = mock_model
            service.model_filename = "test_model.pkl"

            req = PredictionRequest(
                session_id="session-trained-001",
                features=[0.5, 0.8, 1.2]
            )
            resp = service.predict(req)

            self.assertEqual(resp.status, "success")
            self.assertEqual(resp.model_source, "trained_model")
            self.assertFalse(resp.is_placeholder)
            self.assertEqual(resp.feature_dim, 3)
            self.assertEqual(resp.predictions[0].workload, "high")
            self.assertFalse(resp.predictions[0].is_placeholder)
            mock_model.predict.assert_called_once_with([[0.5, 0.8, 1.2]])


if __name__ == "__main__":
    unittest.main()
