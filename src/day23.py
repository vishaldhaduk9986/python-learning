from fastapi import FastAPI, Security, HTTPException, status, Depends, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import secrets
import redis.asyncio as redis
import uvicorn

app = FastAPI()

#-------------------------------
# API Key Authentication
#-------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_API_KEY = "MY_SECRET_QA_KEY"

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key is None or not secrets.compare_digest(api_key, VALID_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key

#-------------------------------
# Initialize Rate Limiter
#-------------------------------
@app.on_event("startup")
async def startup():
    redis_client = await redis.from_url("redis://localhost", encoding="utf-8", decode_responses=True)
    await FastAPILimiter.init(redis_client)

#-------------------------------
# Protected and Rate Limited Endpoint
#-------------------------------
@app.post("/qa", dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def qa_endpoint(
    request: Request, query: str, api_key: str = Security(verify_api_key)
):
    return {"response": f"Answer generated for: {query}"}

#-------------------------------
# Run app
#-------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
