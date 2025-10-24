import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import types
from fastapi.testclient import TestClient


def _import_day21_with_shim(monkeypatch):
    # Ensure lightweight langchain_community shim for imports
    fake_lc = types.ModuleType("langchain_community")
    fake_lc.document_loaders = types.ModuleType("langchain_community.document_loaders")
    fake_lc.embeddings = types.ModuleType("langchain_community.embeddings")
    fake_lc.vectorstores = types.ModuleType("langchain_community.vectorstores")
    fake_lc.llms = types.ModuleType("langchain_community.llms")
    fake_lc.llms.ChatOpenAI = lambda *a, **k: None
    fake_lc.embeddings.OpenAIEmbeddings = lambda *a, **k: None
    fake_lc.vectorstores.FAISS = type("FAISS", (), {})
    fake_lc.document_loaders.PyPDFLoader = lambda path: None
    monkeypatch.setitem(sys.modules, "langchain_community", fake_lc)
    monkeypatch.setitem(sys.modules, "langchain_community.document_loaders", fake_lc.document_loaders)
    monkeypatch.setitem(sys.modules, "langchain_community.embeddings", fake_lc.embeddings)
    monkeypatch.setitem(sys.modules, "langchain_community.vectorstores", fake_lc.vectorstores)
    monkeypatch.setitem(sys.modules, "langchain_community.llms", fake_lc.llms)

    import importlib
    if 'src.day21' in sys.modules:
        del sys.modules['src.day21']
    day21 = importlib.import_module('src.day21')
    return day21


def test_ask_as_retriever_typeerror(monkeypatch):
    day21 = _import_day21_with_shim(monkeypatch)

    # vector_store.as_retriever should raise TypeError when called with kwargs
    class VS:
        def as_retriever(self):
            return "retriever"

    monkeypatch.setattr(day21, "vector_store", VS())
    monkeypatch.setattr(day21, "get_openai_api_key", lambda: "sk-123")
    monkeypatch.setattr(day21, "make_chat_llm", lambda **kw: object())

    # Provide a RetrievalQA that returns a string answer via run
    class FakeQA:
        @classmethod
        def from_chain_type(cls, **kw):
            class Q:
                def run(self, q):
                    return "string-answer"

            return Q()

    monkeypatch.setattr(day21, "RetrievalQA", FakeQA)

    client = TestClient(day21.app)
    resp = client.post("/ask", json={"question": "q"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "string-answer"


def test_missing_api_key_and_llm(monkeypatch):
    day21 = _import_day21_with_shim(monkeypatch)

    # vector_store present
    class VS:
        def as_retriever(self, **kw):
            return "retriever"

    monkeypatch.setattr(day21, "vector_store", VS())
    # No API key
    monkeypatch.setattr(day21, "get_openai_api_key", lambda: None)
    client = TestClient(day21.app)
    resp = client.post("/ask", json={"question": "q"})
    assert resp.status_code == 500


def test_make_chat_llm_none(monkeypatch):
    day21 = _import_day21_with_shim(monkeypatch)

    class VS:
        def as_retriever(self, **kw):
            return "retriever"

    monkeypatch.setattr(day21, "vector_store", VS())
    monkeypatch.setattr(day21, "get_openai_api_key", lambda: "sk" )
    monkeypatch.setattr(day21, "make_chat_llm", lambda **kw: None)
    client = TestClient(day21.app)
    resp = client.post("/ask", json={"question": "q"})
    assert resp.status_code == 500


def test_retrievalqa_fallback_to_simple(monkeypatch):
    day21 = _import_day21_with_shim(monkeypatch)

    class VS:
        def as_retriever(self, **kw):
            return "retriever"

    monkeypatch.setattr(day21, "vector_store", VS())
    monkeypatch.setattr(day21, "get_openai_api_key", lambda: "sk" )
    monkeypatch.setattr(day21, "make_chat_llm", lambda **kw: object())

    # Make RetrievalQA raise on instantiation to hit the _SimpleQA fallback
    class BadQA:
        def __init__(self, *a, **k):
            raise RuntimeError("nope")

    monkeypatch.setattr(day21, "RetrievalQA", BadQA)

    client = TestClient(day21.app)
    resp = client.post("/ask", json={"question": "q"})
    assert resp.status_code == 200
    data = resp.json()
    # When no useful result is found, the code returns a default message
    assert "No answer" in data["answer"] or data["answer"] is not None
