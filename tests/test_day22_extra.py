import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
from fastapi.testclient import TestClient
import src.day22 as day22


def test_log_api_call_writes_file(tmp_path):
    out = tmp_path / "api_logs.txt"
    # Temporarily change working directory so the function writes into tmp_path
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        day22.log_api_call("/some-endpoint", "bob")
        assert out.exists()
        txt = out.read_text()
        assert "/some-endpoint" in txt and "bob" in txt
    finally:
        os.chdir(cwd)
