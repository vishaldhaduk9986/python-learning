import unittest
from fastapi.testclient import TestClient
from src.day4 import app

class TestDay4API(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_hello(self):
        response = self.client.get("/hello")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "Hello, FastAPI!")

    def test_submit_data(self):
        data = {"key": "value"}
        response = self.client.post("/submit", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["key"], "value")

    def test_doc(self):
        response = self.client.get("/doc")
        self.assertEqual(response.status_code, 200)
        self.assertIn("FastAPI Endpoint Documentation", response.json()["documentation"])

if __name__ == "__main__":
    unittest.main()
