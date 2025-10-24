# Python Training Project

This is a starter Python project with the following structure:

- `main.py`: Entry point script
- `src/`: Module folder
- `tests/`: Unit tests

## How to Run

```bash
python main.py
```

## How to Test

```bash
python -m unittest discover tests
```

## day28: Full AI microservice example

`src/day28.py` is a small FastAPI example demonstrating a deployable pattern:

- Endpoints: `/health`, `/ready`, `/predict`
- Authentication: `X-API-KEY` header checked against `SERVICE_API_KEY` env var
- Backend selection: prefers a LangChain-compatible chat LLM via `src.utils.make_chat_llm()`;
  falls back to the Hugging Face Inference API when `HF_INFERENCE_API_TOKEN` or `HF_INFERENCE_API_URL` is provided.

To run locally (dev):

```bash
export SERVICE_API_KEY=dev-key
python -m uvicorn src.day28:app --reload
```

Or run inside Docker — the app is small and designed to avoid loading large models at startup by using hosted inference when configured.
