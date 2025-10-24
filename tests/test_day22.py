import sys
from pathlib import Path
# ensure repo root is on sys.path for imports like `src.xxx`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
import src.day22 as day22


def test_some_endpoint_background_logging(tmp_path, monkeypatch):
    # Replace the file-writing log_api_call with one that writes to a temp file
    out_file = tmp_path / "api_logs_test.txt"

    def fake_log_api_call(endpoint: str, user: str):
        with open(out_file, "a") as f:
            f.write(f"Endpoint: {endpoint} | User: {user}\n")

    monkeypatch.setattr(day22, "log_api_call", fake_log_api_call)

    client = TestClient(day22.app)
    with client:
        # 'user' is parsed as a query param for this endpoint, so pass it in the URL
        resp = client.post("/some-endpoint?user=alice")
        assert resp.status_code == 200
        body = resp.json()
        assert "Request received" in body.get("status", "")

    # After the client context manager exits, background tasks should have run
    contents = out_file.read_text()
    assert "Endpoint: /some-endpoint | User: alice" in contents
