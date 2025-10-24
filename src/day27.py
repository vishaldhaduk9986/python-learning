"""day27.py
-----------
Small FastAPI app exposing two endpoints used by tests:

- POST /qa: question-answering endpoint. Prefers a LangChain-compatible
  Chat LLM via `src.utils.make_chat_llm()` and `invoke_llm_safely`.
  If no LLM is available, returns a lightweight fallback answer.

- POST /sentiment: a tiny sentiment analyzer that uses a rule-based
  scorer (keeps the example dependency-free and fast for unit tests).

This file is designed to be easy to unit-test using FastAPI's
TestClient and monkeypatching.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from src.utils import make_chat_llm, invoke_llm_safely


app = FastAPI(title="day27-qa-sentiment")


class QARequest(BaseModel):
    question: str
    context: Optional[str] = None


class QAResponse(BaseModel):
    answer: str


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    label: str
    score: float


@app.post("/qa", response_model=QAResponse)
def qa(req: QARequest):
    """Answer a question using an available LLM, or a lightweight fallback.

    The function uses `make_chat_llm()` from `src.utils`. Tests can
    monkeypatch `make_chat_llm` to return a fake LLM callable.
    """
    llm = make_chat_llm()
    if llm is not None:
        # normalize invocation via helper
        answer = invoke_llm_safely(llm, req.question)
        return QAResponse(answer=str(answer))

    # Fallback: simple context-aware echo
    if req.context:
        return QAResponse(answer=f"I couldn't reach an LLM. Context: {req.context}")
    return QAResponse(answer="No backend available to answer the question")


def _rule_sentiment(text: str):
    """Very small rule-based sentiment scorer for tests.

    Returns (label, score) where score is between 0 and 1.
    """
    txt = text.lower()
    if "good" in txt or "great" in txt or "happy" in txt:
        return "positive", 0.9
    if "bad" in txt or "sad" in txt or "terrible" in txt:
        return "negative", 0.1
    return "neutral", 0.5


@app.post("/sentiment", response_model=SentimentResponse)
def sentiment(req: SentimentRequest):
    label, score = _rule_sentiment(req.text)
    return SentimentResponse(label=label, score=score)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("src.day27:app", host="0.0.0.0", port=port)
