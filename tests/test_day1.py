import unittest
from src.day1 import main
from io import StringIO
import sys

class TestDay1Main(unittest.TestCase):
    def test_main_prints_hello(self):
        captured_output = StringIO()
        sys.stdout = captured_output
        main()
        sys.stdout = sys.__stdout__
        self.assertIn("Hello, World!", captured_output.getvalue())

if __name__ == "__main__":
    unittest.main()
