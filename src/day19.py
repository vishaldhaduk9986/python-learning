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

# Try to import the OpenAI wrapper compatible with this workspace
try:
    from langchain_openai import OpenAI
except Exception:
    # langchain-openai may not be installed in some environments; fall back
    # to langchain_community if available.
    try:
        from langchain_community.llms import OpenAI  # type: ignore
    except Exception:
        OpenAI = None  # type: ignore

# Initialize FastAPI app
app = FastAPI()


# Define input schema
class QuestionRequest(BaseModel):
    text: str


def get_openai_api_key() -> Optional[str]:
    """Return the OpenAI key from the environment if present."""
    return os.environ.get("OPENAI_API_KEY")


# Prepare your LLM (example with OpenAI key set as env variable)
# The module sets `LLM_AVAILABLE` at import time. Tests mock this behavior
# by reloading the module after injecting fake packages or environment vars.
LLM_AVAILABLE = False
if OpenAI is not None and get_openai_api_key():
    try:
        llm = OpenAI(temperature=0, openai_api_key=get_openai_api_key())
        LLM_AVAILABLE = True
    except Exception:
        llm = None  # type: ignore
else:
    llm = None  # type: ignore


@app.post("/qa")
async def question_answer(req: QuestionRequest):
    """Return an LLM-generated answer if available or a placeholder.

    Note: LLM invocation may differ between LangChain wrappers. The code
    attempts `.invoke()` first then falls back to calling the object.
    """
    # Generate the response using LLM if available, else return a placeholder
    if LLM_AVAILABLE and llm is not None:
        try:
            # Modern langchain models may use __call__ / invoke differently; try a few ways
            if hasattr(llm, "invoke"):
                answer = llm.invoke(req.text)
            else:
                answer = llm(req.text)
        except Exception as e:
            answer = f"LLM invocation failed: {e}"
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
