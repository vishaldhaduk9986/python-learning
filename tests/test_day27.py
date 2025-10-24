import os
import sys

# Make sure repo root is importable like other tests
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient


def test_qa_with_llm(monkeypatch):
    import src.day27 as day27

    client = TestClient(day27.app)

    # Fake LLM that echoes
    def fake_llm(prompt: str):
        return f"answer to: {prompt}"

    monkeypatch.setattr(day27, "make_chat_llm", lambda: fake_llm)

    payload = {"question": "What is testing?", "context": "Unit tests"}
    r = client.post("/qa", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "answer to:" in data["answer"]


def test_qa_fallback(monkeypatch):
    import src.day27 as day27

    client = TestClient(day27.app)

    # No LLM available
    monkeypatch.setattr(day27, "make_chat_llm", lambda: None)

    payload = {"question": "Who?", "context": "Alice and Bob"}
    r = client.post("/qa", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "Context: Alice and Bob" in data["answer"]


def test_sentiment_rules():
    import src.day27 as day27

    client = TestClient(day27.app)

    r = client.post("/sentiment", json={"text": "I had a good day"})
    assert r.status_code == 200
    data = r.json()
    assert data["label"] == "positive"
    assert data["score"] > 0.8
