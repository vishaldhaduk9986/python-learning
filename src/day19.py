"""
day19.py
---------
Lightweight FastAPI example demonstrating an endpoint that proxies
questions to an LLM when available, and returns a graceful placeholder
when not configured.

Developer notes:
- The module attempts to import an OpenAI wrapper that may be provided
  by different langchain packages; this maximizes compatibility across
  environments. If an LLM cannot be imported or the `OPENAI_API_KEY` is
  missing, the endpoint will respond with a friendly message rather
  than raising.
- The module evaluates LLM availability at import time for simplicity.
  If you need dynamic LLM toggling at runtime, move the initialization
  into a startup event or a factory function.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from src.utils import get_openai_api_key, make_chat_llm, invoke_llm_safely

# Initialize FastAPI app
app = FastAPI()


# Define input schema
class QuestionRequest(BaseModel):
    text: str


# Module-level LLM initialization done via shared helper. Tests can
# replace `make_chat_llm` to simulate availability or failure modes.
llm = make_chat_llm(model_name="gpt-4o", temperature=0)
LLM_AVAILABLE = llm is not None


@app.post("/qa")
async def question_answer(req: QuestionRequest):
    """Return an LLM-generated answer if available or a placeholder.

    Uses `invoke_llm_safely` to normalize invocation styles across
    langchain wrappers and test doubles.
    """
    if LLM_AVAILABLE and llm is not None:
        answer = invoke_llm_safely(llm, req.text)
    else:
        answer = "(LLM not available in this environment)"
    return {"question": req.text, "answer": answer}


if __name__ == "__main__":
    # Quick local test without starting the FastAPI server.
    sample = "How many Nobel laureates in Physics have there been as of 2024?"
    if LLM_AVAILABLE and llm is not None:
        try:
            if hasattr(llm, "invoke"):
                resp = llm.invoke(sample)
            else:
                resp = llm(sample)
        except Exception as e:
            print("LLM invocation failed:", e, file=sys.stderr)
            resp = None
    else:
        resp = "(LLM not configured; set OPENAI_API_KEY to use OpenAI)"

    print("Question:", sample)
    print("Answer:", resp)
