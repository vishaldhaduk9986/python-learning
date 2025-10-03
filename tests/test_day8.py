
import unittest
from src.day8 import main
from transformers import pipeline

class TestDay8Sentiment(unittest.TestCase):
    def test_main_coverage(self):
        main()

    def setUp(self):
        self.sentiment = pipeline('sentiment-analysis')

    def test_positive_sentence(self):
        result = self.sentiment(["I'm thrilled with the resultseverything worked perfectly!"])[0]
        self.assertEqual(result['label'], 'POSITIVE')
        self.assertGreater(result['score'], 0.5)

    def test_negative_sentence(self):
        result = self.sentiment(["The instructions were unclear, and I wasted a lot of time."])[0]
        self.assertEqual(result['label'], 'NEGATIVE')
        self.assertGreater(result['score'], 0.5)

    def test_neutral_sentence(self):
        result = self.sentiment(["It's just okay, nothing special but not terrible either."])[0]
        self.assertGreater(result['score'], 0.5)

if __name__ == "__main__":
    unittest.main()
