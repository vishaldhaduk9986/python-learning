"""
Sentiment Analysis FastAPI Service
----------------------------------
This module provides a REST API for sentiment analysis using Hugging Face Transformers.

Endpoints:
    POST /analyze: Analyze sentiment of input text.

Usage:
    - Start the server with: uvicorn src.day10:app --reload
    - Send POST requests to /analyze with JSON body: {"text": "your text"}

Dependencies:
    - fastapi
    - pydantic
    - transformers
"""

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
sentiment_pipe = pipeline("sentiment-analysis")


# Initialize sentiment analysis pipeline
sentiment_pipe = pipeline("sentiment-analysis")

# Create FastAPI app instance
app = FastAPI()

class TextRequest(BaseModel):
    """
    Request model for sentiment analysis.
    Attributes:
        text (str): The input text to analyze.
    """
    text: str

class SentimentResponse(BaseModel):
    """
    Response model for sentiment analysis results.
    Attributes:
        label (str): Sentiment label (e.g., POSITIVE, NEGATIVE).
        score (float): Confidence score for the prediction.
    """
    label: str
    score: float

@app.post("/analyze", response_model=SentimentResponse, summary="Analyze sentiment of text", response_description="Sentiment label and score")
def analyze(request: TextRequest) -> SentimentResponse:
    """
    Analyze the sentiment of the provided text using Hugging Face Transformers.
    Args:
        request (TextRequest): Input text wrapped in a Pydantic model.
    Returns:
        SentimentResponse: Sentiment label and confidence score.
    """
    result = sentiment_pipe(request.text)[0]
    return SentimentResponse(label=result["label"], score=result["score"])
