

"""
day12.py
--------
IMDB Sentiment Analysis Sample Script
------------------------------------
This script loads a sample of IMDB movie reviews and runs sentiment analysis using Hugging Face Transformers.
It prints the first 100 characters of each review and the predicted sentiment label and score.

How to Run:
    python src/day12.py

Dependencies:
    - datasets
    - transformers

Author: [Your Name]
Date: 2025-10-03
"""

from datasets import load_dataset
from transformers import pipeline

def main():
    """
    Loads 5 random IMDB reviews and prints their sentiment analysis results.
    Uses Hugging Face's sentiment-analysis pipeline.
    """
    # Load IMDB dataset
    imdb = load_dataset("imdb")

    # Sample 5 random examples from training set
    sample = imdb["train"].shuffle(seed=42).select(range(5))

    # Load sentiment analysis pipeline
    sentiment = pipeline("sentiment-analysis")

    # Run sentiment analysis on sampled reviews with truncation enabled
    for example in sample:
        text = example['text']
        result = sentiment(text, truncation=True, max_length=512)[0]
        print(f"Review: {text[:100]}...")  # print first 100 chars for readability
        print(f"Sentiment: {result['label']} (score: {result['score']:.4f})\n")

if __name__ == "__main__":
    main()
