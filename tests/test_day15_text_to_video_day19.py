import os
import io
import sys
import contextlib
import runpy
import types
from pathlib import Path

import pytest

from fastapi.testclient import TestClient


def test_day15_get_openai_api_key_and_main_behavior(monkeypatch, tmp_path):
    # Ensure no OPENAI_API_KEY yields exit code 1
    mod_path = 'src.day15'
    # Clear env
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)

    # Run module main and capture stderr
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        # run_module will raise because imports are faked in other tests; just import and call functions
        mod = __import__(mod_path, fromlist=['*'])
        code = mod.main()
    assert code == 1

    # Now set a dummy key and monkeypatch OpenAI usage to avoid network
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')

    class DummyOpenAI:
        def __init__(self, *args, **kwargs):
            pass

    class DummyChain:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, sentence):
            return 'polite: ' + sentence

    # Inject fakes into module attributes
    monkeypatch.setattr('src.day15.OpenAI', DummyOpenAI, raising=False)
    monkeypatch.setattr('src.day15.LLMChain', DummyChain, raising=False)

    # Now calling main should proceed and return 0
    mod = __import__(mod_path, fromlist=['*'])
    rc = mod.main()
    assert rc == 0


def test_day19_fastapi_endpoint_with_no_llm(monkeypatch):
    # Import the app and TestClient
    import importlib

    mod = importlib.import_module('src.day19')
    client = TestClient(mod.app)

    # Ensure environment has no OPENAI key and no OpenAI class
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    # Force LLM availability flags
    monkeypatch.setattr('src.day19.LLM_AVAILABLE', False, raising=False)

    resp = client.post('/qa', json={'text': 'hello'})
    assert resp.status_code == 200
    data = resp.json()
    assert data['question'] == 'hello'
    assert '(LLM not available' in data['answer'] or data['answer'] == '(LLM not available in this environment)'


def test_text_to_video_color_helpers_and_cartoon(monkeypatch, tmp_path):
    # Import helpers
    import importlib

    tt = importlib.import_module('src.text_to_video')

    # Test hex_to_rgb
    assert tt.hex_to_rgb('#000000') == (0, 0, 0)
    assert tt.hex_to_rgb('#ffffff') == (255, 255, 255)

    # Test interpolate_color returns tuple of ints and is between start/end
    c = tt.interpolate_color('#000000', '#ffffff', 0.5)
    assert isinstance(c, tuple) and len(c) == 3
    for v in c:
        assert 0 <= v <= 255

    # Prepare a tiny input file
    p = tmp_path / 'input.txt'
    p.write_text('one two')

    out_video = tmp_path / 'out.mp4'

    # Monkeypatch gTTS to avoid network and file writes
    class DummyTTS:
        def __init__(self, text, lang='en'):
            self.text = text

        def save(self, filename):
            # create a tiny silent mp3 file using bytes; moviepy won't actually be invoked in tests
            with open(filename, 'wb') as f:
                f.write(b'ID3')

    monkeypatch.setattr('src.text_to_video.gTTS', DummyTTS)

    # Monkeypatch ImageSequenceClip and AudioFileClip to lightweight fakes
    class FakeAudioClip:
        def __init__(self, filename):
            self.duration = 0.5

    class FakeClip:
        def __init__(self, frames, durations=None):
            self.frames = frames
            self.durations = durations
            self.audio = None

        def write_videofile(self, path, fps, codec, audio):
            # create an empty file to simulate output
            Path(path).write_text('video')

    monkeypatch.setattr('src.text_to_video.AudioFileClip', FakeAudioClip)
    monkeypatch.setattr('src.text_to_video.ImageSequenceClip', FakeClip)

    # Run create_video_from_text (plain) with cartoon overlay True
    tt.create_video_from_text(str(p), str(out_video), with_cartoon=True)
    assert out_video.exists()

    # Run the cartoon-specific function as well
    out_cartoon = tmp_path / 'cartoon.mp4'
    tt.create_cartoon_video_from_text(str(p), str(out_cartoon))
    assert out_cartoon.exists()
