import importlib
import sys
import types
import runpy
import io
import contextlib

import pytest


def test_import_chat_openai_fallback_to_langchain_llms(monkeypatch):
    # Remove any existing modules
    monkeypatch.delitem(sys.modules, 'langchain_community.llms', raising=False)
    monkeypatch.delitem(sys.modules, 'langchain_openai', raising=False)

    # Create package 'langchain' and submodule 'langchain.llms'
    pkg = types.ModuleType('langchain')
    llms = types.ModuleType('langchain.llms')

    class FinalChat:
        pass

    llms.ChatOpenAI = FinalChat
    monkeypatch.setitem(sys.modules, 'langchain', pkg)
    monkeypatch.setitem(sys.modules, 'langchain.llms', llms)

    mod = importlib.import_module('src.utils')
    importlib.reload(mod)
    assert mod.import_chat_openai() is FinalChat


def test_day13_hf_api_raises_and_local_pipeline(monkeypatch):
    mod = importlib.import_module('src.day13')
    from fastapi.testclient import TestClient

    monkeypatch.setenv('HF_INFERENCE_API_TOKEN', 'fake')

    def fake_raise(text):
        raise RuntimeError('boom')

    monkeypatch.setattr(mod, 'call_hf_inference_api', fake_raise)
    # ensure local pipeline will be used
    monkeypatch.setattr(mod, '_init_local_pipeline', lambda: None)

    class DummySent:
        def __call__(self, text, truncation=True, max_length=512):
            return [{'label': 'MIXED', 'score': 0.5}]

    monkeypatch.setattr(mod, 'sentiment', DummySent(), raising=False)
    client = TestClient(mod.app)
    resp = client.post('/analyze', json={'text': 'x'})
    assert resp.status_code == 200
    assert resp.json()['label'] == 'MIXED'


def test_day21_string_result_and_as_retriever_typeerror(monkeypatch):
    mod = importlib.import_module('src.day21')
    from fastapi.testclient import TestClient

    # vector store with as_retriever raising TypeError to hit except branch
    class VS:
        def as_retriever(self, search_kwargs=None):
            raise TypeError('no kwargs')

    mod.vector_store = VS()

    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    import src.utils as utils_mod
    class FakeLLM:
        pass
    monkeypatch.setattr(utils_mod, 'make_chat_llm', lambda *a, **k: FakeLLM())

    # RetrievalQA that returns a plain string when run
    class PlainQA:
        def run(self, q):
            return 'plain-answer'

    monkeypatch.setattr('src.day21.RetrievalQA', PlainQA, raising=False)

    # Monkeypatch loader/embeddings so upload works
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

    from io import BytesIO
    client = TestClient(mod.app)
    resp = client.post('/upload_pdf', files={'file': ('t.pdf', BytesIO(b'%PDF-1.4'), 'application/pdf')})
    assert resp.status_code == 200

    resp2 = client.post('/ask', json={'question': 'q'})
    assert resp2.status_code == 200
    assert resp2.json()['answer'] == 'plain-answer'


def test_text_to_video_audio_assign_exception(monkeypatch, tmp_path):
    tt = importlib.import_module('src.text_to_video')

    # Prepare an input file
    p = tmp_path / 'in.txt'
    p.write_text('one two')
    out = tmp_path / 'o.mp4'

    # Monkeypatch gTTS and AudioFileClip
    class DummyTTS:
        def __init__(self, text, lang='en'):
            pass
        def save(self, fn):
            open(fn, 'wb').write(b'ID3')

    monkeypatch.setattr('src.text_to_video.gTTS', DummyTTS)

    class FakeAudio:
        def __init__(self, fn):
            self.duration = 0.5

    class FakeClip:
        def __init__(self, frames, durations=None):
            self.frames = frames
        @property
        def audio(self):
            raise AttributeError('cannot assign')

    monkeypatch.setattr('src.text_to_video.AudioFileClip', FakeAudio)
    monkeypatch.setattr('src.text_to_video.ImageSequenceClip', FakeClip)

    # write_videofile may be called; provide a simple implementation on FakeClip
    def write_videofile(self, path, fps, codec, audio):
        open(path, 'w').write('v')

    FakeClip.write_videofile = write_videofile

    tt.create_video_from_text(str(p), str(out), with_cartoon=False)
    assert out.exists()
