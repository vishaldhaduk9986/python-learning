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
