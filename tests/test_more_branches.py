import importlib
import runpy
import sys
import types
import io
import contextlib
from pathlib import Path

import pytest


def test_utils_make_chat_llm_constructor_failure(monkeypatch):
    mod = importlib.import_module('src.utils')
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')

    # Fake ChatOpenAI that raises on construction
    fake = types.ModuleType('langchain_community.llms')

    class Broken:
        def __init__(self, *a, **k):
            raise RuntimeError('construct fail')

    fake.ChatOpenAI = Broken
    monkeypatch.setitem(sys.modules, 'langchain_community.llms', fake)

    # Should return None rather than raising
    assert mod.make_chat_llm() is None


def test_day13_hf_api_dict_and_empty_list(monkeypatch):
    mod = importlib.import_module('src.day13')
    from fastapi.testclient import TestClient

    # Case: HF API returns a dict
    monkeypatch.setenv('HF_INFERENCE_API_TOKEN', 'fake')

    def fake_dict(text):
        return {'label': 'POSITIVE', 'score': 0.99}

    monkeypatch.setattr(mod, 'call_hf_inference_api', fake_dict)
    client = TestClient(mod.app)
    resp = client.post('/analyze', json={'text': 'hi'})
    assert resp.status_code == 200
    assert resp.json()['label'] == 'POSITIVE'

    # Case: HF API returns empty list -> falls back to local pipeline
    def fake_empty(text):
        return []

    monkeypatch.setattr(mod, 'call_hf_inference_api', fake_empty)
    # Monkeypatch _init_local_pipeline and sentiment
    monkeypatch.setattr(mod, '_init_local_pipeline', lambda: None)
    class DummySent:
        def __call__(self, text, truncation=True, max_length=512):
            return [{'label': 'NEGATIVE', 'score': 0.12}]
    monkeypatch.setattr(mod, 'sentiment', DummySent(), raising=False)

    resp2 = client.post('/analyze', json={'text': 'bad'})
    assert resp2.status_code == 200
    assert resp2.json()['label'] == 'NEGATIVE'


def test_day21_result_shapes_and_sources(monkeypatch):
    mod = importlib.import_module('src.day21')
    from fastapi.testclient import TestClient

    # Prepare vector store and retriever
    class FakeVS:
        def as_retriever(self, search_kwargs=None):
            return self

    mod.vector_store = FakeVS()

    # Provide API key and fake LLM
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    import src.utils as utils_mod
    class FakeLLM:
        pass
    monkeypatch.setattr(utils_mod, 'make_chat_llm', lambda *a, **k: FakeLLM())

    # Case 1: RetrievalQA.from_chain_type exists and returns chain with run returning dict
    class QAChain:
        def run(self, q):
            return {'answer': 'ok', 'source_documents': [types.SimpleNamespace(metadata={'page': 2}, page_content='hello')]}

    class FakeRetrievalQA:
        @classmethod
        def from_chain_type(cls, llm, retriever, chain_type, return_source_documents):
            return QAChain()

    monkeypatch.setattr('src.day21.RetrievalQA', FakeRetrievalQA, raising=False)

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

    client = TestClient(mod.app)
    import io as _io
    files = {'file': ('t.pdf', _io.BytesIO(b'%PDF-1.4'), 'application/pdf')}
    resp = client.post('/upload_pdf', files=files)
    assert resp.status_code == 200

    resp2 = client.post('/ask', json={'question': 'what'})
    assert resp2.status_code == 200
    data = resp2.json()
    assert 'answer' in data
    assert isinstance(data.get('sources', []), list)


def test_text_to_video_cli_calls(monkeypatch, tmp_path):
    # Monkeypatch functions so CLI path doesn't actually write large files
    mod = importlib.import_module('src.text_to_video')

    called = {}

    def fake_create(input_path, out, with_cartoon=False):
        called['created'] = (input_path, out, with_cartoon)

    monkeypatch.setattr('src.text_to_video.create_video_from_text', fake_create, raising=False)
    monkeypatch.setattr('src.text_to_video.create_cartoon_video_from_text', fake_create, raising=False)

    # Create a small input file and call the monkeypatched functions directly
    p = tmp_path / 'in.txt'
    p.write_text('hello world')
    out = tmp_path / 'out.mp4'

    mod.create_video_from_text(str(p), str(out), with_cartoon=False)
    assert 'created' in called

    called.clear()
    out2 = tmp_path / 'out2.mp4'
    mod.create_cartoon_video_from_text(str(p), str(out2))
    assert 'created' in called


def test_day3_import_and_file_write(monkeypatch, tmp_path):
    # Run day3 in a temporary cwd and fake requests.get
    import requests as _requests

    class FakeResp:
        def json(self):
            return {'main': {'temp': 25, 'humidity': 50}}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('src.day3.requests.get', lambda url: FakeResp())

    # run module
    runpy.run_module('src.day3', run_name='__main__')
    # weather.json should be written
    p = tmp_path / 'weather.json'
    assert p.exists()
