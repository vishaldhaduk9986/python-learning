
import unittest
from src.day9 import main
from transformers import pipeline

class TestDay9Summarization(unittest.TestCase):
    def test_main_coverage(self):
        main()

    def setUp(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.article = (
            "Hugging Face, Inc. is an American company based in New York City that develops computation tools for building applications using machine learning. "
            "It is most notable for its transformers library built for natural language processing applications and its platform that allows users to share machine learning models and datasets and showcase their work. "
            "The company was founded in 2016 by French entrepreneurs Clment Delangue, Julien Chaumond, and Thomas Wolf in New York City, originally as a company that developed a chatbot app targeted at teenagers. "
            "After open sourcing the model behind the chatbot, the company pivoted to focus on being a platform for machine learning."
        )

    def test_summary_length(self):
        summary = self.summarizer(self.article, max_length=50, min_length=25, do_sample=False)
        text = summary[0]['summary_text']
        self.assertTrue(25 <= len(text.split()) <= 50)

    def test_summary_content(self):
        summary = self.summarizer(self.article, max_length=50, min_length=25, do_sample=False)
        text = summary[0]['summary_text']
        self.assertIn("Hugging Face", text)
        keywords = ["machine learning", "chatbot", "transformers", "New York"]
        self.assertTrue(any(kw in text for kw in keywords), f"Summary missing expected keywords: {keywords}")

if __name__ == "__main__":
    unittest.main()
