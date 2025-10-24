import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import types
from fastapi.testclient import TestClient


def _import_day21_with_shim(monkeypatch):
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


def test_answer_key_and_metadata_object(monkeypatch):
    day21 = _import_day21_with_shim(monkeypatch)

    # Setup vector store retriever
    class FakeVS:
        def as_retriever(self, **kw):
            return "retriever"

    monkeypatch.setattr(day21, "vector_store", FakeVS())
    monkeypatch.setattr(day21, "get_openai_api_key", lambda: "sk-123")
    monkeypatch.setattr(day21, "make_chat_llm", lambda **kw: object())

    # Create a doc whose metadata is an object with attribute `page`
    class MetaObj:
        def __init__(self, page):
            self.page = page

    class Doc:
        def __init__(self, content, page):
            self.page_content = content
            self.metadata = MetaObj(page)

    # RetrievalQA returns a dict with 'answer' key and source_documents having metadata object
    class FakeQA:
        @classmethod
        def from_chain_type(cls, **kw):
            class Q:
                def run(self, q):
                    return {"answer": "ans-key", "source_documents": [Doc('text here', 7)]}

            return Q()

    monkeypatch.setattr(day21, "RetrievalQA", FakeQA)

    client = TestClient(day21.app)
    resp = client.post("/ask", json={"question": "q"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "ans-key"
    assert isinstance(data.get("sources"), list)
    assert data["sources"][0]["page"] == 7


def test_callable_qa_chain(monkeypatch):
    day21 = _import_day21_with_shim(monkeypatch)

    class FakeVS:
        def as_retriever(self, **kw):
            return "retriever"

    monkeypatch.setattr(day21, "vector_store", FakeVS())
    monkeypatch.setattr(day21, "get_openai_api_key", lambda: "sk-123")
    monkeypatch.setattr(day21, "make_chat_llm", lambda **kw: object())

    # Return an object that's callable but has no .run to hit the callable path
    class CallableQA:
        @classmethod
        def from_chain_type(cls, **kw):
            class C:
                def __call__(self, payload):
                    return {"result": "call-result", "source_documents": []}

            return C()

    monkeypatch.setattr(day21, "RetrievalQA", CallableQA)

    client = TestClient(day21.app)
    resp = client.post("/ask", json={"question": "q"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "call-result"
