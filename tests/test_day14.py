import unittest
from fastapi.testclient import TestClient
from src.day14 import app

class TestDay14API(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_sentiment_positive(self):
        response = self.client.post("/sentiment", json={"text": "I love this product!"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["label"], ["POSITIVE", "NEGATIVE"])
        self.assertGreaterEqual(data["score"], 0.0)
        self.assertLessEqual(data["score"], 1.0)

    def test_summary(self):
        long_text = "This is a long text that should be summarized. " * 10
        response = self.client.post("/summary", json={"text": long_text})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(isinstance(data["summary"], str))
        self.assertGreater(len(data["summary"]), 0)

if __name__ == "__main__":
    unittest.main()
