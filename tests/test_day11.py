import unittest
import json
import os
from transformers import pipeline
import src.day11 as day11


class TestDay11Performance(unittest.TestCase):
    def test_performance_logs_created(self):
        # Run the script to generate both logs
        import src.day11
        # Check DistilBERT log
        distilbert_log = "inference_performance_distilbert.json"
        bert_log = "inference_performance_bert.json"
        self.assertTrue(os.path.exists(distilbert_log))
        self.assertTrue(os.path.exists(bert_log))
        with open(distilbert_log) as f:
            data = json.load(f)
        self.assertEqual(data["model"], "distilbert-base-uncased-finetuned-sst-2-english")
        self.assertEqual(data["num_sentences"], 50)
        with open(bert_log) as f:
            data = json.load(f)
        self.assertEqual(data["model"], "bert-base-uncased")
        self.assertEqual(data["num_sentences"], 50)
        # Clean up
        os.remove(distilbert_log)
        os.remove(bert_log)

if __name__ == "__main__":
    unittest.main()
