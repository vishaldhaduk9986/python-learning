import os
import sys

# Ensure repo root is importable
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient


def test_qa_llm_raises(monkeypatch):
    import src.day27 as day27

    class BadLLM:
        def invoke(self, prompt):
            raise RuntimeError("boom at runtime")

    monkeypatch.setattr(day27, "make_chat_llm", lambda: BadLLM())

    client = TestClient(day27.app)
    r = client.post("/qa", json={"question": "Will it fail?"})
    assert r.status_code == 200
    data = r.json()
    assert "LLM invocation failed" in data["answer"]


def test_qa_missing_question_returns_422():
    import src.day27 as day27

    client = TestClient(day27.app)
    r = client.post("/qa", json={})
    assert r.status_code == 422


def test_sentiment_invalid_type_returns_422():
    import src.day27 as day27

    client = TestClient(day27.app)
    r = client.post("/sentiment", json={"text": 123})
    assert r.status_code == 422
