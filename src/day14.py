"""
Text Analysis API
=================

This FastAPI application provides endpoints for sentiment analysis and text summarization using Hugging Face models.

Endpoints:
    - POST /sentiment: Analyze sentiment of input text.
    - POST /summary: Generate a summary for input text.

Usage:
    - Start the server with: uvicorn src.day14:app --reload
    - Access Swagger docs at: http://127.0.0.1:8000/docs

Dependencies:
    - fastapi
    - pydantic
    - transformers

OpenAPI Tags:
    - Sentiment: Endpoints for sentiment analysis
    - Summary: Endpoints for text summarization
"""

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize pipelines (loading cached models)
sentiment_pipe = pipeline("sentiment-analysis")
summary_pipe = pipeline("summarization", model="facebook/bart-large-cnn")

app = FastAPI(
    title="Text Analysis API",
    description="API for sentiment analysis and text summarization using Hugging Face models.",
    version="1.0.0",
    openapi_tags=[
        {"name": "Sentiment", "description": "Endpoints for sentiment analysis"},
        {"name": "Summary", "description": "Endpoints for text summarization"},
    ],
)

class TextRequest(BaseModel):
    """
    Request model for text input.
    Attributes:
        text (str): The input text to analyze or summarize.
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

class SummaryResponse(BaseModel):
    """
    Response model for text summarization results.
    Attributes:
        summary (str): Generated summary of the input text.
    """
    summary: str

@app.post(
    "/sentiment",
    response_model=SentimentResponse,
    summary="Analyze sentiment of text",
    description="Analyze the sentiment of the given input text and return label with confidence score.",
    response_description="Sentiment label and confidence score",
    tags=["Sentiment"]
)
def analyze_sentiment(request: TextRequest) -> SentimentResponse:
    """
    Analyze the sentiment of the provided text using Hugging Face Transformers.
    Args:
        request (TextRequest): Input text wrapped in a Pydantic model.
    Returns:
        SentimentResponse: Sentiment label and confidence score.
    """
    result = sentiment_pipe(request.text, truncation=True, max_length=512)[0]
    return SentimentResponse(label=result["label"], score=result["score"])

@app.post(
    "/summary",
    response_model=SummaryResponse,
    summary="Summarize input text",
    description="Generate a 2-line summary for the given input text.",
    response_description="Summary text",
    tags=["Summary"]
)
def summarize_text(request: TextRequest) -> SummaryResponse:
    """
    Summarize the provided text using Hugging Face Transformers.
    Args:
        request (TextRequest): Input text wrapped in a Pydantic model.
    Returns:
        SummaryResponse: Summary string of the input text.
    """
    summary_result = summary_pipe(request.text, max_length=50, min_length=20, do_sample=False)
    return SummaryResponse(summary=summary_result[0]['summary_text'])
