# Deploying this FastAPI microservice (src/day13.py)

This project exposes a FastAPI app in `src/day13.py` with a POST `/analyze` endpoint that uses Hugging Face Transformers for sentiment analysis.

Notes up-front

- Models like `bert`/`distilbert` will be downloaded at runtime. That can be large (100s of MB). Free hosts have limited disk and memory and may not allow large builds.
- To reduce memory use pick a small model (e.g., `distilbert-base-uncased-finetuned-sst-2-english`) by setting the environment variable `MODEL_NAME` before starting and updating code to use it.

Recommended free hosts (shortlist)

- Render (free web-service) — easy for Python apps, supports `requirements.txt` and `Procfile`.
- Railway (free usage tier) — simple git-based deploys; may sleep inactive services.
- Fly.io (free tier) — lightweight VMs, good control, but requires creating a small Dockerfile.
- PythonAnywhere — quick for small, CPU-light apps (may not allow long downloads).

Quick deploy (Render or Railway)

1. Create a new GitHub repo and push this project there.
2. On Render: Create > New Web Service > Connect GitHub repo. Select Python. Build command: `pip install -r requirements.txt` (Render auto-detects). Start command: `uvicorn src.day13:app --host 0.0.0.0 --port $PORT`.
3. On Railway: Create new project > Deploy from GitHub. Set start command (or Procfile will be used): `uvicorn src.day13:app --host 0.0.0.0 --port $PORT`.

Fly.io (Docker) — useful if you need more control

1. Install flyctl and sign up.
2. Create `Dockerfile` (minimal example):
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt /app/
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . /app
   CMD ["uvicorn", "src.day13:app", "--host", "0.0.0.0", "--port", "8080"]
3. Then `fly launch` and `fly deploy`.

Environment variables and model size

- If you want to force a smaller model, change `src/day13.py` to load pipeline with a MODEL_NAME env var, e.g.:
  model = os.environ.get('MODEL_NAME', 'distilbert-base-uncased-finetuned-sst-2-english')
  sentiment = pipeline('sentiment-analysis', model=model)
- Add a `requirements.txt` (provided) so hosts install the correct packages.

Verifying the service after deploy

- Curl the endpoint (replace <URL> with your deployed host):

  curl -X POST "https://<URL>/analyze" -H "Content-Type: application/json" -d '{"text": "I love this movie!"}'

Troubleshooting

- Large model downloads causing build timeouts: pre-download model locally and use a smaller model, or switch to a host that supports larger images (paid tiers).
- Memory errors: try a smaller model, or switch to a hosted inference API (Hugging Face Inference API) and call it from your app.

If you want, I can:

- Add code to `src/day13.py` to read `MODEL_NAME` from env and fallback to a smaller default.
- Add a Dockerfile tailored for Fly.io.
- Create a GitHub Actions workflow to auto-deploy to Render or Railway.

Using Hugging Face Inference API (avoid local model downloads)

- If you don't want to download models during deployment, use the Hugging Face Inference API. Set the environment variable `HF_INFERENCE_API_TOKEN` to your Hugging Face API token.
- Optionally set `HF_INFERENCE_API_URL` to a custom inference URL (defaults to `https://api-inference.huggingface.co/models/<MODEL_NAME>`).
- When `HF_INFERENCE_API_TOKEN` is present the app will call the hosted inference API instead of loading a local model. This avoids large runtime downloads and is recommended for free hosts.

Example env vars on Render/Railway:

- HF_INFERENCE_API_TOKEN = <your_hf_token_here>
- MODEL_NAME = distilbert-base-uncased-finetuned-sst-2-english

Docker-based deploy (recommended to avoid OOM on small hosts)

- The repository includes `Dockerfile` and `requirements.deploy.txt` which install only the lightweight runtime dependencies (no `torch` or `transformers`).
- The Docker image expects `HF_INFERENCE_API_TOKEN` to be set in the runtime environment so the app will use the Hugging Face Inference API instead of loading local models.

Build and run locally:

```bash
# build
docker build -t sentiment-service:latest .

# run (replace <your_token> and expose port as needed)
docker run -e HF_INFERENCE_API_TOKEN=<your_token> -e MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english -p 8000:8000 sentiment-service:latest
```

Deploy to Fly.io (example)

1. Install flyctl and login.
2. `fly launch` (choose a region and app name).
3. Set secrets on fly: `fly secrets set HF_INFERENCE_API_TOKEN=<your_token> MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english`.
4. `fly deploy`.
