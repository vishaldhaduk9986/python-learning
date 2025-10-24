import sys
from pathlib import Path
# ensure repo root is on sys.path for imports like `src.xxx`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
import src.day26 as day26
import pytest


def test_root_access():
    client = TestClient(day26.app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json().get("message")


def test_qa_endpoint_unauthorized():
    client = TestClient(day26.app)
    # 'query' is a query param for this endpoint
    resp = client.post("/qa?query=x")
    assert resp.status_code == 401


def test_qa_endpoint_authorized():
    client = TestClient(day26.app)
    headers = {"X-API-Key": day26.VALID_API_KEY}
    resp = client.post("/qa?query=how+are+you", headers=headers)
    assert resp.status_code == 200
    assert "Answer generated for: how are you" in resp.json().get("response")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I love this product", "positive"),
        ("I hate this", "negative"),
        ("", "neutral"),
        ("Just ok", "neutral"),
    ],
)
def test_sentiment_analyze(text, expected):
    client = TestClient(day26.app)
    # 'text' is parsed as a query parameter on this endpoint
    resp = client.post("/sentiment", params={"text": text})
    assert resp.status_code == 200
    assert resp.json().get("sentiment") == expected
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
VALID_API_KEY = "MY_SECRET_KEY"

# ----- Tests for /qa endpoint -----

def test_qa_success():
    response = client.post(
        "/qa",
        headers={"X-API-Key": VALID_API_KEY},
        json={"query": "What is FastAPI?"}
    )
    assert response.status_code == 200
    assert response.json() == {"response": "Answer generated for: What is FastAPI?"}

def test_qa_missing_api_key():
    response = client.post("/qa", json={"query": "test"})
    assert response.status_code == 401
    assert response.json() == {"message": "Invalid or missing API Key"}

def test_qa_validation_error():
    response = client.post(
        "/qa",
        headers={"X-API-Key": VALID_API_KEY},
        json={}  # Missing required 'query'
    )
    assert response.status_code == 422
    assert "detail" in response.json()

# ----- Tests for /sentiment endpoint -----

def test_sentiment_positive():
    response = client.post("/sentiment", json={"text": "I love this!"})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"

def test_sentiment_negative():
    response = client.post("/sentiment", json={"text": "I hate bugs."})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "negative"

def test_sentiment_missing_text():
    response = client.post("/sentiment", json={})
    assert response.status_code == 422  # validation error for missing required field
    assert "detail" in response.json()
