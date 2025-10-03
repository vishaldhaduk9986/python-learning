from fastapi import FastAPI, Request
from pydantic import BaseModel

import logging

logger = logging.getLogger("uvicorn")

app = FastAPI()

class HelloResponse(BaseModel):
    message: str

class SubmitRequest(BaseModel):
    key: str

@app.get("/hello", response_model=HelloResponse, summary="Get greeting message", response_description="A greeting string")
async def hello():
    """
    Returns a greeting message.
    """
    return HelloResponse(message="Hello, FastAPI!")

@app.post("/submit", response_model=SubmitRequest, summary="Submit data", response_description="Echoes submitted data")
async def submit_data(request: Request):
    """
    Accepts JSON data and returns it.
    """
    data = await request.json()
    return data

@app.get("/doc")
async def get_doc():
    """Return API documentation as a string."""
    doc = (
        "# FastAPI Endpoint Documentation\n"
        "\n"
        "## Endpoints\n"
        "- GET /hello: Returns a greeting message.\n"
        "- POST /submit: Accepts JSON data and returns it.\n"
        "\n"
        "## Usage\n"
        "Start the server: uvicorn src.day4:app --reload\n"
        "Access docs: http://127.0.0.1:8000/docs\n"
    )
    return {"documentation": doc}
    
@app.get("/", include_in_schema=False)
async def root():
    """Lightweight health endpoint to confirm the service and routes."""
    return {"service": "day4", "status": "ok"}

@app.on_event("startup")
async def startup_event():
    logger.info("Starting day4 FastAPI app with routes: /hello, /submit, /doc")



