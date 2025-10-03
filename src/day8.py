

"""
day8.py
------
Sentiment Analysis Example Script
--------------------------------
This script demonstrates how to use Hugging Face Transformers to perform sentiment analysis on a list of custom sentences.
It prints the sentiment label and score for each sentence.

How to Run:
    python src/day8.py

Dependencies:
    - transformers

Author: [Your Name]
Date: 2025-10-03
"""

from transformers import pipeline

def main():
    """
    Loads the sentiment analysis pipeline and analyzes a list of custom sentences.
    Prints the sentiment result for each sentence.
    """
    # Load the sentiment analysis pipeline
    sentiment = pipeline('sentiment-analysis')

    # Define custom sentences
    sentences = [
        "I'm thrilled with the results—everything worked perfectly!",
        "The instructions were unclear, and I wasted a lot of time.",
        "It's just okay, nothing special but not terrible either."
    ]

    # Run sentiment analysis
    results = sentiment(sentences)

    # Print results
    for sent, res in zip(sentences, results):
        print(f"Sentence: {sent}")
        print(f"Result: {res}\n")

if __name__ == "__main__":
    main()
