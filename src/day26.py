import logging
from logging.config import dictConfig
from fastapi import FastAPI, Security, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.exceptions import RequestValidationError
import secrets

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "app_errors.log",
            "formatter": "default",
        }
    },
    "loggers": {
        "app": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
}

dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("app")

app = FastAPI()

# API Key Authentication Setup
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_API_KEY = "MY_SECRET_KEY"

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key is None or not secrets.compare_digest(api_key, VALID_API_KEY):
        logger.error(f"Unauthorized access attempt with API key: {api_key}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key

# Exception handler for request validation errors (invalid requests)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    raw_body = await request.body()
    logger.error(f"Validation error for request: {raw_body.decode('utf-8', errors='ignore')} - Errors: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )

# Exception handler for HTTP exceptions (e.g. unauthorized)
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP error {exc.status_code} at {request.url}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )

# Example secured API endpoint that requires the API key
@app.post("/qa")
async def qa_endpoint(query: str, api_key: str = Security(verify_api_key)):
    logger.info(f"QA query received: {query}")
    response_text = f"Answer generated for: {query}"
    return {"response": response_text}

# Root endpoint logs info on access
@app.get("/")
async def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to the secure FastAPI app with logging"}

@app.post("/sentiment")
async def sentiment_analyze(text: str):
    # Dummy sentiment logic for illustration
    if not text:
        sentiment = "neutral"
    elif "love" in text.lower():
        sentiment = "positive"
    elif "hate" in text.lower():
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return {"sentiment": sentiment}


