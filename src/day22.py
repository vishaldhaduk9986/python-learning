from fastapi import FastAPI, BackgroundTasks
from datetime import datetime

app = FastAPI()

def log_api_call(endpoint: str, user: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] Endpoint: {endpoint} | User: {user}\n"
    with open("api_logs.txt", "a") as f:
        f.write(log_entry)

@app.post("/some-endpoint")
async def some_endpoint(
    user: str, background_tasks: BackgroundTasks
):
    # Schedule background logging
    background_tasks.add_task(log_api_call, "/some-endpoint", user)
    return {"status": "Request received, logging to file in background"}
