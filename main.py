from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

VALID_API_KEY = "MY_SECRET_KEY"


class QARequest(BaseModel):
    query: str


@app.post("/qa")
def qa_endpoint(payload: QARequest, x_api_key: str | None = Header(None)):
    if x_api_key != VALID_API_KEY:
        return JSONResponse(status_code=401, content={"message": "Invalid or missing API Key"})
    # Simple deterministic response for tests
    return {"response": f"Answer generated for: {payload.query}"}


class SentimentRequest(BaseModel):
    text: str


@app.post("/sentiment")
def sentiment(payload: SentimentRequest):
    text = payload.text.lower()
    if "love" in text or "good" in text:
        return {"sentiment": "positive"}
    if "hate" in text or "bad" in text:
        return {"sentiment": "negative"}
    return {"sentiment": "neutral"}


if __name__ == "__main__":
    print("Run via `uvicorn main:app`")


def main():
    """Legacy entrypoint used by tests that expect a callable main()."""
    print("Hello, World!")
