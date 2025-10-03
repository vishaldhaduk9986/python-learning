import unittest
from fastapi.testclient import TestClient
from src.day6 import app, Task

class TestDay6API(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.task_data = {"description": "Test Task"}
        response = self.client.post("/task", json=self.task_data)
        self.task_id = response.json()["id"]

    def test_create_task(self):
        data = {"description": "Another Task"}
        response = self.client.post("/task", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["description"], "Another Task")

    def test_read_tasks(self):
        response = self.client.get("/tasks")
        self.assertEqual(response.status_code, 200)
        tasks = response.json()
        self.assertTrue(any(task["description"] == "Test Task" for task in tasks))

if __name__ == "__main__":
    unittest.main()
