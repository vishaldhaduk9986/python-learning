
"""
day13.py
---------
FastAPI Sentiment Analysis Microservice
--------------------------------------
This module provides a REST API for sentiment analysis using Hugging Face Transformers.
It exposes a single POST endpoint `/analyze` that accepts text and returns the sentiment label and score.

Features:
- Uses FastAPI for API definition and validation
- CORS enabled for local frontend development
- Hugging Face Transformers pipeline for inference

How to Run:
    python src/day13.py
    # or
    uvicorn src.day13:app --reload

Example Request:
    curl -X POST "http://127.0.0.1:8000/analyze" -H "Content-Type: application/json" -d '{"text": "I love this movie!"}'

Author: [Your Name]
Date: 2025-10-03
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

# Initialize sentiment analysis pipeline globally (model cached)
sentiment = pipeline("sentiment-analysis")

app = FastAPI()

# Enable CORS - adjust origins as needed
origins = [
    "http://localhost",
    "http://localhost:3000",  # React/Vue frontend default port
    # add more if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # Allow specified origins
    allow_credentials=True,
    allow_methods=["*"],        # Allow all HTTP methods
    allow_headers=["*"],        # Allow all headers
)

class TextRequest(BaseModel):
    """Request model for sentiment analysis input text."""
    text: str

class SentimentResponse(BaseModel):
    """Response model for sentiment analysis result."""
    label: str
    score: float


@app.post(
    "/analyze",
    response_model=SentimentResponse,
    summary="Analyze sentiment of text",
    response_description="Sentiment label and score"
)

def analyze(request: TextRequest) -> SentimentResponse:
    """
    Analyze the sentiment of the provided text using Hugging Face Transformers.

    Args:
        request (TextRequest): Input text wrapped in a Pydantic model.

    Returns:
        SentimentResponse: Sentiment label and confidence score.
    """
    result = sentiment(request.text, truncation=True, max_length=512)[0]
    return SentimentResponse(label=result["label"], score=result["score"])


if __name__ == "__main__":
    import uvicorn
    # Run with live-reload for local development
    uvicorn.run("src.day13:app", host="127.0.0.1", port=8000, reload=True)

#curl -X POST "http://127.0.0.1:8000/analyze" -H "Content-Type: application/json" -d '{"text": "I love this movie!"}'

