import unittest
from fastapi.testclient import TestClient
from src.day7 import app, Book, BookUpdate

class TestDay7API(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Add a book for testing
        self.book_data = {"title": "Test Book", "author": "Author"}
        response = self.client.post("/books", json=self.book_data)
        self.book_id = response.json()["id"]

    def test_get_books(self):
        response = self.client.get("/books")
        self.assertEqual(response.status_code, 200)
        books = response.json()
        self.assertTrue(any(book["title"] == "Test Book" for book in books))

    def test_add_book(self):
        data = {"title": "Another Book", "author": "Another Author"}
        response = self.client.post("/books", json=data)
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["title"], "Another Book")

    def test_update_book(self):
        update = {"title": "Updated Title"}
        response = self.client.patch(f"/books/{self.book_id}", json=update)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["title"], "Updated Title")

    def test_delete_book(self):
        response = self.client.delete(f"/books/{self.book_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["title"], "Test Book")
        # Confirm deletion
        response = self.client.get(f"/books/{self.book_id}")
        self.assertEqual(response.status_code, 404)

if __name__ == "__main__":
    unittest.main()
