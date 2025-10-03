
"""
day9.py
-------
Text Summarization Example Script
--------------------------------
This script demonstrates how to use Hugging Face Transformers to summarize a long text using the BART model.
It prints a concise summary of a sample article.

How to Run:
	python src/day9.py

Dependencies:
	- transformers

Author: [Your Name]
Date: 2025-10-03
"""

from transformers import pipeline

def main():
	"""
	Loads the BART summarization pipeline and summarizes a sample article.
	Prints the summary to stdout.
	"""
	# Load the BART summarization pipeline
	summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

	# Example: Hugging Face Wikipedia snippet (replace with your own article or text as needed)
	article = (
		"Hugging Face, Inc. is an American company based in New York City that develops computation tools for building applications using machine learning. "
		"It is most notable for its transformers library built for natural language processing applications and its platform that allows users to share machine learning models and datasets and showcase their work. "
		"The company was founded in 2016 by French entrepreneurs Clément Delangue, Julien Chaumond, and Thomas Wolf in New York City, originally as a company that developed a chatbot app targeted at teenagers. "
		"After open sourcing the model behind the chatbot, the company pivoted to focus on being a platform for machine learning."
	)

	# Summarize with a target of ~2 sentences
	summary = summarizer(article, max_length=50, min_length=25, do_sample=False)
	print(summary[0]['summary_text'])

if __name__ == "__main__":
	main()
