from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_edge.server import app


class ApiContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_home_and_local_assets(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("桥观", response.text)
        self.assertNotIn("https://", response.text)
        self.assertEqual(self.client.get("/assets/app.js").status_code, 200)

    @patch("bridge_edge.server.available_models", return_value=[])
    @patch("bridge_edge.server.hardware", return_value={"cuda_available": False, "device": "CPU"})
    def test_health_does_not_pretend_a_model_is_ready(self, *_):
        body = self.client.get("/api/health").json()
        self.assertEqual(body["status"], "model_not_ready")
        self.assertFalse(body["external_api_required"])
        self.assertEqual(len(body["classes"]), 5)

    def test_artifact_names_are_restricted(self):
        for url in ("/api/artifacts/not-an-id/annotated.jpg", "/api/artifacts/" + "a" * 32 + "/private.txt"):
            self.assertEqual(self.client.get(url).status_code, 404)

    def test_unknown_history_id(self):
        self.assertEqual(self.client.get("/api/history/invalid").status_code, 404)

    def test_invalid_pagination(self):
        self.assertEqual(self.client.get("/api/history?limit=1000").status_code, 400)

    @patch("bridge_edge.server._sample_map", return_value={})
    def test_missing_sample(self, _):
        self.assertEqual(self.client.get("/api/samples/no.jpg").status_code, 404)


if __name__ == "__main__":
    unittest.main()
