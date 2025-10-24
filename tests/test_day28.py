import os
import sys
import types
import pytest

# Ensure repo root is on sys.path so `import src.*` works in focused tests
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def make_fake_llm(response_text: str):
    # Return a simple callable object that returns response_text
    def _call(prompt: str):
        return response_text

    return _call


def test_predict_with_llm(monkeypatch):
    import src.day28 as day28

    # Ensure the service key matches the module default
    api_key = getattr(day28, "SERVICE_API_KEY")

    # Monkeypatch make_chat_llm to return a callable LLM
    monkeypatch.setattr(day28, "make_chat_llm", lambda: make_fake_llm("llm-reply"))

    req = day28.PredictRequest(prompt="Hello")
    resp = day28.predict(req, x_api_key=api_key)

    assert resp.model == "ChatOpenAI"
    assert "llm-reply" in resp.output


def test_predict_hf_fallback(monkeypatch):
    import src.day28 as day28

    api_key = getattr(day28, "SERVICE_API_KEY")

    # Force LLM not available
    monkeypatch.setattr(day28, "make_chat_llm", lambda: None)
    # Ensure module thinks HF is available
    monkeypatch.setattr(day28, "HF_TOKEN", "fake-token")

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResponse([{"generated_text": "hf-reply"}])

    monkeypatch.setattr(day28, "requests", types.SimpleNamespace(post=fake_post))

    req = day28.PredictRequest(prompt="Hello HF")
    resp = day28.predict(req, x_api_key=api_key)

    assert resp.model == "hf-inference"
    assert "hf-reply" in resp.output


def test_predict_unauthorized(monkeypatch):
    import src.day28 as day28

    # Wrong key should raise HTTPException
    req = day28.PredictRequest(prompt="whoami")
    with pytest.raises(Exception):
        day28.predict(req, x_api_key="wrong-key")
