
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
import requests
import os
import logging

logger = logging.getLogger("uvicorn")

# Read model name from env to allow using lighter models on free hosts
# Default to a reasonably small, sentiment-finetuned DistilBERT model
MODEL_NAME = os.environ.get(
    "MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english"
)

# If HF Inference API token is set, prefer calling the hosted inference API
# Read token early so we don't load a local model unnecessarily
HF_API_TOKEN = os.environ.get("HF_INFERENCE_API_TOKEN")
HF_API_URL = os.environ.get("HF_INFERENCE_API_URL") or f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

# Local pipeline is lazily initialized only when needed (to avoid high memory usage at startup)
sentiment = None


def _init_local_pipeline():
    """Initialize the local transformers pipeline and cache it in the module scope."""
    global sentiment
    if sentiment is not None:
        return
    try:
        sentiment = pipeline("sentiment-analysis", model=MODEL_NAME)
        logger.info(f"Loaded sentiment model: {MODEL_NAME}")
    except Exception:
        logger.exception("Failed to load specified model, falling back to default pipeline.")
        sentiment = pipeline("sentiment-analysis")


def call_hf_inference_api(text: str):
    """Call Hugging Face Inference API for text classification.

    Returns the first result dict or raises an exception on failure.
    """
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text}
    resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()

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
    # If HF API token is provided, call the hosted inference API to avoid local model loads
    if HF_API_TOKEN:
        try:
            api_result = call_hf_inference_api(request.text)
            # API returns a list of dicts for classification
            if isinstance(api_result, list) and api_result:
                r = api_result[0]
                return SentimentResponse(label=r.get("label"), score=float(r.get("score", 0.0)))
            # If API returns a dict or unexpected format, attempt to extract
            if isinstance(api_result, dict):
                # e.g. {'label': 'POSITIVE', 'score': 0.98}
                return SentimentResponse(label=api_result.get("label"), score=float(api_result.get("score", 0.0)))
        except Exception:
            logger.exception("HF Inference API call failed; falling back to local pipeline.")

    # Local pipeline fallback - initialize lazily to avoid using memory on small hosts
    _init_local_pipeline()
    result = sentiment(request.text, truncation=True, max_length=512)[0]
    return SentimentResponse(label=result["label"], score=result["score"])


if __name__ == "__main__":
    import uvicorn
    # Run with live-reload for local development
    uvicorn.run("src.day13:app", host="127.0.0.1", port=8000, reload=True)

#curl -X POST "http://127.0.0.1:8000/analyze" -H "Content-Type: application/json" -d '{"text": "I love this movie!"}'

