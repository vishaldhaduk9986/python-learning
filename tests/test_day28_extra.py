import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
import src.day28 as day28


def test_predict_with_llm(monkeypatch):
    monkeypatch.setattr(day28, "require_api_key", lambda k: None)
    monkeypatch.setattr(day28, "make_chat_llm", lambda **kw: object())
    monkeypatch.setattr(day28, "invoke_llm_safely", lambda llm, p: "hello world")

    client = TestClient(day28.app)
    resp = client.post("/predict", json={"prompt": "tell me"}, headers={"X-API-KEY": "dev-key"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "ChatOpenAI"


def test_predict_hf_fallback_list(monkeypatch):
    monkeypatch.setattr(day28, "require_api_key", lambda k: None)
    monkeypatch.setattr(day28, "make_chat_llm", lambda **kw: None)

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return [{"generated_text": "from hf"}]

    monkeypatch.setattr(day28, "HF_TOKEN", "token")
    monkeypatch.setattr(day28, "requests", type("R", (), {"post": lambda *a, **k: FakeResp()}))

    client = TestClient(day28.app)
    resp = client.post("/predict", json={"prompt": "hi"}, headers={"X-API-KEY": "dev-key"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"].startswith("hf")


def test_predict_hf_fallback_dict_and_failure(monkeypatch):
    # HF returns a dict-shaped response
    monkeypatch.setattr(day28, "require_api_key", lambda k: None)
    monkeypatch.setattr(day28, "make_chat_llm", lambda **kw: None)

    class FakeResp2:
        def raise_for_status(self):
            return None

        def json(self):
            return {"generated_text": "dict-hf"}

    monkeypatch.setattr(day28, "HF_TOKEN", "token")
    monkeypatch.setattr(day28, "requests", type("R", (), {"post": lambda *a, **k: FakeResp2()}))

    client = TestClient(day28.app)
    resp = client.post("/predict", json={"prompt": "hi"}, headers={"X-API-KEY": "dev-key"})
    assert resp.status_code == 200
    assert resp.json()["model"].startswith("hf")

    # Simulate HF error -> 503
    class BadResp:
        def raise_for_status(self):
            raise RuntimeError("bad")

    monkeypatch.setattr(day28, "requests", type("R2", (), {"post": lambda *a, **k: BadResp()}))
    resp2 = client.post("/predict", json={"prompt": "hi"}, headers={"X-API-KEY": "dev-key"})
    assert resp2.status_code == 503


def test_predict_unauthorized():
    client = TestClient(day28.app)
    resp = client.post("/predict", json={"prompt": "x"})
    assert resp.status_code == 401
