import os
import io
import json
import types
import importlib
import sys
from pathlib import Path

import pytest

from fastapi.testclient import TestClient


def test_day13_hf_api_path_and_local_fallback(monkeypatch, tmp_path):
    mod = importlib.import_module('src.day13')

    # Case 1: HF API token present and API returns a list
    monkeypatch.setenv('HF_INFERENCE_API_TOKEN', 'fake-token')
    fake_resp = [{'label': 'POSITIVE', 'score': 0.95}]

    def fake_call(text):
        return fake_resp

    monkeypatch.setattr(mod, 'call_hf_inference_api', fake_call)

    client = TestClient(mod.app)
    resp = client.post('/analyze', json={'text': 'I love it'})
    assert resp.status_code == 200
    data = resp.json()
    assert data['label'] == 'POSITIVE'
    # score should be a float probability in [0, 1]
    assert isinstance(data['score'], float)
    assert 0.0 <= data['score'] <= 1.0

    # Case 2: HF API raises -> local pipeline fallback. Monkeypatch _init_local_pipeline and sentiment
    def fake_call_raise(text):
        raise RuntimeError('api fail')

    monkeypatch.setenv('HF_INFERENCE_API_TOKEN', 'fake-token')
    monkeypatch.setattr(mod, 'call_hf_inference_api', fake_call_raise)

    # Monkeypatch local pipeline initializer and sentiment callable
    class DummySentiment:
        def __call__(self, text, truncation=True, max_length=512):
            return [{'label': 'NEGATIVE', 'score': 0.12}]

    monkeypatch.setattr(mod, '_init_local_pipeline', lambda: None)
    monkeypatch.setattr(mod, 'sentiment', DummySentiment(), raising=False)

    resp2 = client.post('/analyze', json={'text': 'bad'})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2['label'] == 'NEGATIVE'


def test_day19_llm_available_branch(monkeypatch):
    mod = importlib.import_module('src.day19')
    # Create a dummy llm that supports __call__
    class DummyLLM:
        def __init__(self, *args, **kwargs):
            # accept any init args used by the real OpenAI wrapper
            pass

        def __call__(self, text):
            return 'llm-says-' + text

    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    # Ensure importing langchain_openai picks up our DummyLLM class
    fake_mod = types.ModuleType('langchain_openai')
    fake_mod.OpenAI = DummyLLM
    monkeypatch.setitem(sys.modules, 'langchain_openai', fake_mod)
    # Force a fresh import so module-level LLM_AVAILABLE is recomputed
    if 'src.day19' in sys.modules:
        del sys.modules['src.day19']
    mod = importlib.import_module('src.day19')

    client = TestClient(mod.app)
    resp = client.post('/qa', json={'text': 'hi'})
    assert resp.status_code == 200
    data = resp.json()
    assert data['answer'] == 'llm-says-hi'


def test_day21_upload_and_ask(monkeypatch, tmp_path):
    mod = importlib.import_module('src.day21')
    client = TestClient(mod.app)

    # Monkeypatch PyPDFLoader to return a small doc list
    class DummyDoc:
        def __init__(self, content):
            self.page_content = content

    class FakeLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return [DummyDoc('page1 content'), DummyDoc('page2 content')]

    monkeypatch.setattr('src.day21.PyPDFLoader', FakeLoader, raising=False)

    # Monkeypatch splitter
    class FakeSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=0):
            pass

        def split_documents(self, docs):
            return docs

    monkeypatch.setattr('src.day21.CharacterTextSplitter', FakeSplitter, raising=False)

    # Fake embeddings and FAISS
    class FakeEmbeddings:
        def __init__(self):
            pass

    class FakeFAISS:
        @classmethod
        def from_documents(cls, docs, embeddings):
            inst = types.SimpleNamespace()
            inst.docs = docs
            def as_retriever():
                return inst
            inst.as_retriever = lambda : inst
            def similarity_search(query, k=3):
                return docs[:k]
            inst.similarity_search = similarity_search
            return inst

    monkeypatch.setattr('src.day21.OpenAIEmbeddings', FakeEmbeddings, raising=False)
    monkeypatch.setattr('src.day21.FAISS', FakeFAISS, raising=False)

    # Monkeypatch OpenAI to dummy (used when building QA chain)
    class DummyLLM:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, text):
            return 'answer:' + text

    monkeypatch.setattr('src.day21.OpenAI', DummyLLM, raising=False)

    # Fake RetrievalQA to return a predictable answer
    class FakeRetrievalQA:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, q):
            return 'retrieved: ' + q

    monkeypatch.setattr('src.day21.RetrievalQA', FakeRetrievalQA, raising=False)

    # Create a small fake PDF file content and upload
    pdf_bytes = b'%PDF-1.4 fake pdf content'
    files = {'file': ('test.pdf', io.BytesIO(pdf_bytes), 'application/pdf')}
    resp = client.post('/upload_pdf', files=files)
    assert resp.status_code == 200
    assert resp.json().get('msg')

    # Now call ask
    resp2 = client.post('/ask', json={'question': 'what'})
    assert resp2.status_code == 200
    data = resp2.json()
    assert 'answer' in data
