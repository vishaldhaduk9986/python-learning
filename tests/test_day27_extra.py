import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
import src.day27 as day27


def test_qa_with_llm(monkeypatch):
    # monkeypatch make_chat_llm to return a fake llm and invoke_llm_safely to produce an answer
    monkeypatch.setattr(day27, "make_chat_llm", lambda **kw: object())
    monkeypatch.setattr(day27, "invoke_llm_safely", lambda llm, prompt: "generated-answer")

    client = TestClient(day27.app)
    resp = client.post("/qa", json={"question": "hi"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "generated-answer"


def test_qa_fallback_with_context(monkeypatch):
    # simulate no LLM available
    monkeypatch.setattr(day27, "make_chat_llm", lambda **kw: None)
    client = TestClient(day27.app)
    resp = client.post("/qa", json={"question": "q", "context": "ctx"})
    assert resp.status_code == 200
    assert "Context: ctx" in resp.json()["answer"]


def test_sentiment_rules():
    client = TestClient(day27.app)
    resp = client.post("/sentiment", json={"text": "I feel great"})
    assert resp.status_code == 200
    assert resp.json()["label"] == "positive"
