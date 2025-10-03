
import unittest
from src.day12 import main
from datasets import load_dataset
from transformers import pipeline

class TestDay12Sentiment(unittest.TestCase):
    def test_main_coverage(self):
        main()

    def setUp(self):
        self.imdb = load_dataset("imdb")
        self.sentiment = pipeline("sentiment-analysis")

    def test_sample_sentiment(self):
        sample = self.imdb["train"].shuffle(seed=42).select(range(5))
        for example in sample:
            text = example['text']
            result = self.sentiment(text, truncation=True, max_length=512)[0]
            self.assertIn(result['label'], ["POSITIVE", "NEGATIVE"])
            self.assertGreaterEqual(result['score'], 0.0)
            self.assertLessEqual(result['score'], 1.0)

if __name__ == "__main__":
    unittest.main()
