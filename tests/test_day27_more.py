import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
import src.day27 as day27


def test_sentiment_negative_and_neutral():
    client = TestClient(day27.app)
    r1 = client.post("/sentiment", json={"text": "This is terrible"})
    assert r1.status_code == 200
    assert r1.json()["label"] == "negative"

    r2 = client.post("/sentiment", json={"text": "meh"})
    assert r2.status_code == 200
    assert r2.json()["label"] == "neutral"
