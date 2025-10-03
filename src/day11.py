

"""
day11.py

Performance Benchmarking for Sentiment Analysis Models
----------------------------------------------------
This script benchmarks inference speed for two Hugging Face models:
    - DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
    - BERT (bert-base-uncased)

It runs sentiment analysis on 50 sample sentences and logs total and average inference time to JSON files for each model.

Usage:
    python src/day11.py

Output:
    inference_performance_distilbert.json
    inference_performance_bert.json
"""

import time
import json
from transformers import pipeline


def log_performance(model_name, sentences, output_file):
    """
    Runs sentiment analysis inference on a list of sentences using the specified model.
    Logs total and average inference time to a JSON file.

    Args:
        model_name (str): Hugging Face model name for sentiment analysis.
        sentences (List[str]): List of sentences to analyze.
        output_file (str): Path to output JSON file for performance log.

    Returns:
        None
    """
    sentiment = pipeline("sentiment-analysis", model=model_name)
    start_time = time.time()
    results = [sentiment(sentence)[0] for sentence in sentences]
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(sentences)
    performance_log = {
        "model": model_name,
        "num_sentences": len(sentences),
        "total_inference_time_sec": total_time,
        "average_inference_time_per_sentence_sec": avg_time
    }
    with open(output_file, "w") as f:
        json.dump(performance_log, f, indent=4)
    print(f"Inference complete for {model_name}. Performance logged to {output_file}.")


# Sample sentences for benchmarking (5 unique, repeated to make 50)
sentences = [
    "I love this product!",
    "This is awful and I hate it.",
    "The best experience I've had.",
    "Worst purchase I've ever made.",
    "It's okay, not great.",
] * 10  # Make 50 sentences total


# Run benchmarking for both models
log_performance("distilbert-base-uncased-finetuned-sst-2-english", sentences, "inference_performance_distilbert.json")
log_performance("bert-base-uncased", sentences, "inference_performance_bert.json")
