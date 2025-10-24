import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
import src.day13 as day13


def test_analyze_with_hf_list_response(monkeypatch):
    monkeypatch.setattr(day13, "HF_API_TOKEN", "fake-token")

    def fake_api(text: str):
        return [{"label": "POSITIVE", "score": 0.99}]

    monkeypatch.setattr(day13, "call_hf_inference_api", fake_api)

    client = TestClient(day13.app)
    resp = client.post("/analyze", json={"text": "I love it"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] == "POSITIVE"
    assert abs(data["score"] - 0.99) < 1e-6


def test_analyze_with_hf_dict_response(monkeypatch):
    monkeypatch.setattr(day13, "HF_API_TOKEN", "fake-token")

    def fake_api(text: str):
        return {"label": "NEGATIVE", "score": 0.12}

    monkeypatch.setattr(day13, "call_hf_inference_api", fake_api)

    client = TestClient(day13.app)
    resp = client.post("/analyze", json={"text": "bad"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] == "NEGATIVE"
    assert abs(data["score"] - 0.12) < 1e-6


def test_analyze_hf_fails_fallback_to_local(monkeypatch):
    # Simulate HF token present but API call raising an exception
    monkeypatch.setattr(day13, "HF_API_TOKEN", "fake-token")

    def raise_api(text: str):
        raise RuntimeError("network error")

    monkeypatch.setattr(day13, "call_hf_inference_api", raise_api)

    # Provide a cheap local sentiment implementation
    def fake_init():
        day13.sentiment = lambda txt, **kw: [{"label": "NEUTRAL", "score": 0.5}]

    monkeypatch.setattr(day13, "_init_local_pipeline", fake_init)

    client = TestClient(day13.app)
    resp = client.post("/analyze", json={"text": "meh"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] == "NEUTRAL"


def test_analyze_local_only(monkeypatch):
    # Ensure no HF token so local pipeline is used
    monkeypatch.setattr(day13, "HF_API_TOKEN", None)

    def fake_init():
        day13.sentiment = lambda txt, **kw: [{"label": "POSITIVE", "score": 0.75}]

    monkeypatch.setattr(day13, "_init_local_pipeline", fake_init)

    client = TestClient(day13.app)
    resp = client.post("/analyze", json={"text": "nice"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] == "POSITIVE"
    assert abs(data["score"] - 0.75) < 1e-6
import unittest
from fastapi.testclient import TestClient
from src.day13 import app

class TestDay13API(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_analyze_positive(self):
        response = self.client.post("/analyze", json={"text": "I love this movie!"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["label"], ["POSITIVE", "NEGATIVE"])
        self.assertGreaterEqual(data["score"], 0.0)
        self.assertLessEqual(data["score"], 1.0)

    def test_analyze_negative(self):
        response = self.client.post("/analyze", json={"text": "This was a terrible experience."})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["label"], ["POSITIVE", "NEGATIVE"])
        self.assertGreaterEqual(data["score"], 0.0)
        self.assertLessEqual(data["score"], 1.0)

    def test_cors_headers(self):
        response = self.client.options("/analyze", headers={"Origin": "http://localhost:3000"})
        self.assertIn("access-control-allow-origin", response.headers)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://localhost:3000")

if __name__ == "__main__":
    unittest.main()
