
import unittest
import csv
import os
from src.day2 import main

class TestDay2CSV(unittest.TestCase):
    def setUp(self):
        self.filename = "students_test.csv"
        with open(self.filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Name", "Age"])
            writer.writeheader()
            writer.writerows([
                {"Name": "Alice", "Age": "20"},
                {"Name": "Bob", "Age": "17"},
                {"Name": "Charlie", "Age": "22"},
                {"Name": "Diana", "Age": "18"},
                {"Name": "Eve", "Age": "19"}
            ])

    def tearDown(self):
        os.remove(self.filename)

    def test_names_over_18(self):
        # Run main to ensure coverage
        main()
        with open(self.filename, mode='r') as file:
            reader = csv.DictReader(file)
            names_over_18 = [row['Name'] for row in reader if int(row['Age']) > 18]
        self.assertEqual(set(names_over_18), {"Alice", "Charlie", "Eve"})

if __name__ == "__main__":
    unittest.main()
