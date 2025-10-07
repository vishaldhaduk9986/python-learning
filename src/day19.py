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
    return os.environ.get("OPENAI_API_KEY")


# Prepare your LLM (example with OpenAI key set as env variable)
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
