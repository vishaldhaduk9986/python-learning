import unittest
from fastapi.testclient import TestClient
from src.day10 import app

class TestDay10API(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_analyze_positive(self):
        response = self.client.post("/analyze", json={"text": "I love this product!"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["label"], ["POSITIVE", "NEGATIVE"])
        self.assertGreater(data["score"], 0.5)

    def test_analyze_negative(self):
        response = self.client.post("/analyze", json={"text": "This is the worst experience ever."})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["label"], ["POSITIVE", "NEGATIVE"])
        self.assertGreater(data["score"], 0.5)

    def test_analyze_empty(self):
        response = self.client.post("/analyze", json={"text": ""})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["label"], ["POSITIVE", "NEGATIVE"])
        self.assertGreaterEqual(data["score"], 0.0)

if __name__ == "__main__":
    unittest.main()
