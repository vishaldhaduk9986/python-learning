import importlib
import runpy
import sys
import types
import io
import contextlib

import pytest


def test_import_chat_openai_priority_and_fallbacks(monkeypatch):
    # Prepare a fake community module
    comm = types.ModuleType('langchain_community.llms')
    class CommChat:
        pass
    comm.ChatOpenAI = CommChat
    monkeypatch.setitem(sys.modules, 'langchain_community.llms', comm)

    mod = importlib.import_module('src.utils')
    assert mod.import_chat_openai() is CommChat

    # Remove community and add langchain_openai module with ChatOpenAI attr
    monkeypatch.delitem(sys.modules, 'langchain_community.llms', raising=False)
    fake_openai = types.ModuleType('langchain_openai')
    class OpenAIChat:
        pass
    fake_openai.ChatOpenAI = OpenAIChat
    monkeypatch.setitem(sys.modules, 'langchain_openai', fake_openai)
    # reload utils to pick up new import order
    importlib.reload(mod)
    assert mod.import_chat_openai() is OpenAIChat


def test_make_chat_llm_and_invoke(monkeypatch):
    mod = importlib.import_module('src.utils')

    # No API key -> None
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    assert mod.make_chat_llm(openai_api_key=None) is None

    # Provide API key and a fake ChatOpenAI class
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    fake_mod = types.ModuleType('langchain_community.llms')

    class Dummy:
        def __init__(self, *args, **kwargs):
            self._args = args
            self._kwargs = kwargs

        def __call__(self, prompt):
            return 'called:' + prompt

        def invoke(self, prompt):
            return 'invoked:' + prompt

    fake_mod.ChatOpenAI = Dummy
    monkeypatch.setitem(sys.modules, 'langchain_community.llms', fake_mod)

    llm = mod.make_chat_llm()
    assert llm is not None
    # invocation via helper
    assert mod.invoke_llm_safely(llm, 'x') in ('invoked:x', 'called:x')


def test_invoke_llm_safely_exceptions(monkeypatch):
    mod = importlib.import_module('src.utils')

    class Bad:
        def __call__(self, text):
            raise RuntimeError('fail')

    assert 'LLM invocation failed' in mod.invoke_llm_safely(Bad(), 'x')


def test_day19_main_prints_with_and_without_llm(monkeypatch, capsys):
    # Ensure make_chat_llm returns a dummy that will be invoked in __main__
    import src.utils as utils_mod

    class Dummy:
        def __call__(self, text):
            return 'resp:' + text

    monkeypatch.setattr(utils_mod, 'make_chat_llm', lambda *a, **k: Dummy())

    # run module as __main__ and capture output
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_module('src.day19', run_name='__main__')
    out = buf.getvalue()
    assert 'Question:' in out and 'Answer:' in out

    # Now simulate no LLM
    monkeypatch.setattr(utils_mod, 'make_chat_llm', lambda *a, **k: None)
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        runpy.run_module('src.day19', run_name='__main__')
    out2 = buf2.getvalue()
    assert 'LLM not configured' in out2 or 'Answer:' in out2


def test_day21_ask_variants(monkeypatch):
    mod = importlib.import_module('src.day21')
    from fastapi.testclient import TestClient

    # Prepare a fake vector_store with retriever that supports search_kwargs
    class FakeVS1:
        def as_retriever(self, search_kwargs=None):
            return self

    mod.vector_store = FakeVS1()

    # Missing API key -> 500
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    client = TestClient(mod.app)
    resp = client.post('/ask', json={'question': 'x'})
    # API key missing should return a 500 HTTP error (server configuration)
    assert resp.status_code == 500

    # Now provide API key and fake LLM/QA chain behaviors
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')

    class FakeLLM:
        def __init__(self, *a, **k):
            pass

    # monkeypatch make_chat_llm to return our fake
    import src.utils as utils_mod
    monkeypatch.setattr(utils_mod, 'make_chat_llm', lambda *a, **k: FakeLLM())

    # Fake RetrievalQA factory
    class FakeQA:
        def __init__(self, *a, **k):
            pass

        def run(self, q):
            return 'answer for ' + q

    monkeypatch.setattr('src.day21.RetrievalQA', FakeQA, raising=False)

    client = TestClient(mod.app)
    # Upload a fake PDF first to set vector_store properly
    import io as _io
    files = {'file': ('test.pdf', _io.BytesIO(b'%PDF-1.4'), 'application/pdf')}
    # Monkeypatch loaders/embeddings so upload works
    class FakeLoader:
        def __init__(self, path):
            pass

        def load(self):
            class D:
                def __init__(self):
                    self.page_content = 'p'
                    self.metadata = {'page': 1}

            return [D()]

    monkeypatch.setattr('src.day21.PyPDFLoader', FakeLoader, raising=False)
    class FakeEmb:
        def __init__(self):
            pass
    class FakeFAISS:
        @classmethod
        def from_documents(cls, docs, embeddings):
            inst = types.SimpleNamespace()
            inst.as_retriever = lambda search_kwargs=None: inst
            inst.similarity_search = lambda q, k=3: docs
            return inst
    monkeypatch.setattr('src.day21.OpenAIEmbeddings', FakeEmb, raising=False)
    monkeypatch.setattr('src.day21.FAISS', FakeFAISS, raising=False)

    resp = client.post('/upload_pdf', files=files)
    assert resp.status_code == 200

    # Now ask
    resp2 = client.post('/ask', json={'question': 'what'})
    assert resp2.status_code == 200
    assert 'answer' in resp2.json()
