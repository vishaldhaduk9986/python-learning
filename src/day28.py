"""day28.py
-----------
Full AI Microservice example combining FastAPI, Hugging Face hosted
inference fallback, and LangChain-compatible ChatOpenAI usage.

Features:
- FastAPI app with simple API-key header authentication
- /health and /ready endpoints for container health checks
- /predict endpoint that prefers a local/langchain ChatOpenAI via
  `src.utils.make_chat_llm` but falls back to the Hugging Face Inference
  API when available to avoid loading heavy models in-container.
- Structured logging and simple metrics counters (in-memory)

Environment variables:
- SERVICE_API_KEY : required to call /predict (for this example)
- OPENAI_API_KEY  : optional, used by make_chat_llm if present
- HF_INFERENCE_API_TOKEN : optional, used to call hosted HF inference
- HF_INFERENCE_API_URL : optional override for HF endpoint

This file is intentionally self-contained and small; it demonstrates a
deployable microservice pattern suitable for Docker + cloud. Tests can
monkeypatch `src.utils.make_chat_llm` and `requests.post` to avoid
network calls.
"""
from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Any

import requests
from fastapi import FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel

from src.utils import make_chat_llm, invoke_llm_safely, get_openai_api_key


logger = logging.getLogger("day28")
if not logger.handlers:
    # basic config for CLI/container runs
    logging.basicConfig(level=os.environ.get("DAY28_LOG_LEVEL", "INFO"))


SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY", "dev-key")
HF_TOKEN = os.environ.get("HF_INFERENCE_API_TOKEN")
HF_URL = os.environ.get("HF_INFERENCE_API_URL")


app = FastAPI(title="day28-full-ai-microservice", version="0.1")


class PredictRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 128


class PredictResponse(BaseModel):
    model: str
    output: str
    meta: Optional[Dict[str, Any]] = None


# Simple in-memory metrics (not for production)
METRICS: Dict[str, int] = {"requests": 0, "llm_calls": 0, "hf_calls": 0}


def require_api_key(x_api_key: Optional[str]):
    if not x_api_key or x_api_key != SERVICE_API_KEY:
        logger.warning("Unauthorized access attempt")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    # In a real service check DB, vectorstore, etc. For now always ready.
    return {"ready": True}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, x_api_key: Optional[str] = Header(None)):
    """Return a text completion/answer.

    The endpoint requires a service API key in the `X-API-KEY` header.
    It prefers a LangChain-compatible ChatOpenAI provided by
    `make_chat_llm`, and falls back to calling the Hugging Face
    Inference API when `HF_INFERENCE_API_TOKEN` is set.
    """
    require_api_key(x_api_key)
    METRICS["requests"] += 1

    prompt = req.prompt

    # Prefer local/langchain LLMs (testable via monkeypatching make_chat_llm)
    llm = make_chat_llm()
    if llm is not None:
        METRICS["llm_calls"] += 1
        out = invoke_llm_safely(llm, prompt)
        return PredictResponse(model="ChatOpenAI", output=str(out))

    # Fallback to HF Inference API when available
    if HF_TOKEN or HF_URL:
        METRICS["hf_calls"] += 1
        token = HF_TOKEN
        url = HF_URL or os.environ.get("HF_INFERENCE_API_URL") or "https://api-inference.huggingface.co/models/gpt2"
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": req.max_tokens}}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            # HF inference may return a list of generations or dicts
            if isinstance(data, list) and data:
                text = data[0].get("generated_text") or data[0].get("text") or str(data[0])
            elif isinstance(data, dict):
                text = data.get("generated_text") or data.get("text") or str(data)
            else:
                text = str(data)
            return PredictResponse(model="hf-inference", output=text, meta={"raw": data})
        except Exception as e:
            logger.exception("HF inference call failed")
            raise HTTPException(status_code=503, detail="Inference backend failed")

    # No backend available
    logger.error("No LLM backend configured")
    raise HTTPException(status_code=503, detail="No LLM backend available")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("src.day28:app", host="0.0.0.0", port=port, log_level="info")
