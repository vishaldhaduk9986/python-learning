import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import types
from fastapi.testclient import TestClient
import io


def make_fake_doc(text, page=1):
    class D:
        def __init__(self, text, page):
            self.page_content = text
            self.metadata = {"page": page}

    return D(text, page)


def test_upload_pdf_and_index(monkeypatch, tmp_path):
    # Ensure lightweight langchain_community shim is present only for this test
    fake_lc = types.ModuleType("langchain_community")
    fake_lc.document_loaders = types.ModuleType("langchain_community.document_loaders")
    fake_lc.document_loaders.PyPDFLoader = lambda path: None
    fake_lc.embeddings = types.ModuleType("langchain_community.embeddings")
    fake_lc.vectorstores = types.ModuleType("langchain_community.vectorstores")
    fake_lc.llms = types.ModuleType("langchain_community.llms")
    fake_lc.llms.ChatOpenAI = lambda *a, **k: None
    fake_lc.embeddings.OpenAIEmbeddings = lambda *a, **k: None
    fake_lc.vectorstores.FAISS = type("FAISS", (), {})
    monkeypatch.setitem(sys.modules, "langchain_community", fake_lc)
    monkeypatch.setitem(sys.modules, "langchain_community.document_loaders", fake_lc.document_loaders)
    monkeypatch.setitem(sys.modules, "langchain_community.embeddings", fake_lc.embeddings)
    monkeypatch.setitem(sys.modules, "langchain_community.vectorstores", fake_lc.vectorstores)
    monkeypatch.setitem(sys.modules, "langchain_community.llms", fake_lc.llms)

    # Import the module after shimming
    import importlib
    if 'src.day21' in sys.modules:
        del sys.modules['src.day21']
    day21 = importlib.import_module('src.day21')

    # Patch PyPDFLoader to return predictable docs
    class FakeLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return [make_fake_doc("hello world", page=1), make_fake_doc("more", page=2)]

    monkeypatch.setattr(day21, "PyPDFLoader", FakeLoader)

    # Patch splitter to return the same docs
    class FakeSplitter:
        def __init__(self, **kw):
            pass

        def split_documents(self, docs):
            return docs

    monkeypatch.setattr(day21, "CharacterTextSplitter", FakeSplitter)

    # Patch embeddings & FAISS to create a simple vector_store
    class FakeEmb:
        def __init__(self):
            pass

    class FakeFAISS:
        @classmethod
        def from_documents(cls, docs, embeddings):
            class VS:
                def as_retriever(self, **kw):
                    return "retriever"

            return VS()

    monkeypatch.setattr(day21, "OpenAIEmbeddings", FakeEmb)
    monkeypatch.setattr(day21, "FAISS", FakeFAISS)

    client = TestClient(day21.app)

    data = {"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")}
    resp = client.post("/upload_pdf", files=data)
    assert resp.status_code == 200
    assert "uploaded and indexed" in resp.json().get("msg")


def test_ask_no_index(monkeypatch):
    # Provide a lightweight langchain_community shim for this test as well
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
    # Ensure vector_store is None
    monkeypatch.setattr(day21, "vector_store", None)
    client = TestClient(day21.app)
    resp = client.post("/ask", json={"question": "hi"})
    assert resp.status_code == 400


def test_ask_with_index_and_llm(monkeypatch):
    # Ensure shim available for imports
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

    # Setup a fake vector_store whose as_retriever returns a retriever object
    class FakeVS:
        def as_retriever(self, **kw):
            return "retriever"

    monkeypatch.setattr(day21, "vector_store", FakeVS())

    # Ensure get_openai_api_key returns a key
    monkeypatch.setattr(day21, "get_openai_api_key", lambda: "sk-123")

    # Fake LLM
    class FakeLLM:
        def __call__(self, *args, **kwargs):
            return "LLM ANSWER"

    monkeypatch.setattr(day21, "make_chat_llm", lambda **kw: FakeLLM())

    # Fake RetrievalQA with from_chain_type
    class FakeQA:
        @classmethod
        def from_chain_type(cls, **kw):
            class Q:
                def run(self, q):
                    return {"result": "Answer from QA", "source_documents": [make_fake_doc("page text", 5)]}

            return Q()

    monkeypatch.setattr(day21, "RetrievalQA", FakeQA)

    client = TestClient(day21.app)
    resp = client.post("/ask", json={"question": "what"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Answer from QA"
    assert isinstance(body.get("sources"), list)
