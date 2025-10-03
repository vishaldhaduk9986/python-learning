import unittest
from fastapi.testclient import TestClient
from src.day5 import app, User

class TestDay5API(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.user_data = {"name": "TestUser", "age": 30}
        response = self.client.post("/user", json=self.user_data)
        self.assertEqual(response.status_code, 201)

    def test_create_user(self):
        data = {"name": "AnotherUser", "age": 25}
        response = self.client.post("/user", json=data)
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["name"], "AnotherUser")

    def test_get_user(self):
        response = self.client.get("/user/TestUser")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["name"], "TestUser")

    def test_get_user_not_found(self):
        response = self.client.get("/user/UnknownUser")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "User not found")

if __name__ == "__main__":
    unittest.main()
