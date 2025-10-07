# Deploying this repository (day4 and day13)

This repository contains multiple small FastAPI apps under `src/`.

- `src/day4.py` — lightweight example API with endpoints: `/` (health), `/hello`, `/submit`, `/doc`.
- `src/day13.py` — sentiment-analysis API (`/analyze`) which can run locally with `transformers`/`torch` or use the Hugging Face Inference API when `HF_INFERENCE_API_TOKEN` is set.

Files useful for deployment

- `Procfile` — already present and points to `uvicorn src.day4:app --host 0.0.0.0 --port $PORT` by default (used by Render for Python services).
- `requirements.deploy.txt` — lightweight runtime deps (no heavy ML libs). Good for low-memory hosts.
- `requirements.txt` — full requirements (includes transformers/torch) for local heavy inference if needed.
- `Dockerfile` — builds a small image using `requirements.deploy.txt` and runs `uvicorn src.day4:app` by default.

Important notes

- On low-memory free hosts (512Mi), do not install `torch`/`transformers` — use `requirements.deploy.txt` and the Hugging Face Inference API (set `HF_INFERENCE_API_TOKEN`).
- Choose which app you want Render to run: `day4` (lightweight) or `day13` (sentiment). The `Procfile`/start command controls which one is exposed.

Deploy to Render — quick guide

Render supports two main approaches: a Python service (Render builds from your repo) or a Docker service (you provide a Dockerfile). I’ll show both.

Option A — Python service (no Docker) — easiest

1. Push your repository to GitHub and connect it to Render.
2. On Render: New -> Web Service -> Connect Repository -> choose branch.
3. Environment: select **Python**.
4. Build command: leave blank (Render auto-detects) or explicitly set:
   pip install -r requirements.deploy.txt
   (Use `requirements.deploy.txt` to avoid installing heavy ML libs.)
5. Start command: set to the app you want to expose:
   - For the lightweight service: `uvicorn src.day4:app --host 0.0.0.0 --port $PORT`
   - For the sentiment API: `uvicorn src.day13:app --host 0.0.0.0 --port $PORT` (only if you install full `requirements.txt` or plan to use HF Inference API).
6. Environment -> Add Environment Variables:
   - If using HF hosted inference: `HF_INFERENCE_API_TOKEN` = your_token
   - Optionally: `MODEL_NAME` = distilbert-base-uncased-finetuned-sst-2-english
7. Create the service and Deploy.

Option B — Docker service (recommended for reproducibility)

1. Push repository to GitHub.
2. On Render: New -> Web Service -> Connect repo and choose **Docker**.
3. Render will build the Dockerfile in your repo. The included `Dockerfile` uses `requirements.deploy.txt` and runs `uvicorn src.day4:app` by default.
4. Set environment variables on the Render service (HF token, MODEL_NAME if needed).
5. Deploy and test.

Why choose Docker on Render here?

- You control exactly what is installed and avoid surprises from the platform resolver.
- The Dockerfile in this repo is already tuned to be small (no torch/transformers). When you set `HF_INFERENCE_API_TOKEN`, the app uses the hosted HF API so the container stays small and memory friendly.

Testing your deployed service

- Health (root):
  curl https://<YOUR-SERVICE>.onrender.com/
  Expected: {"service":"day4","status":"ok"} (if `day4` is the exposed app)
- Lightweight endpoints (`day4`):
  curl https://<YOUR-SERVICE>.onrender.com/hello
  curl -X POST https://<YOUR-SERVICE>.onrender.com/submit -H "Content-Type: application/json" -d '{"key":"value"}'
- Sentiment endpoint (`day13`):
  curl -X POST https://<YOUR-SERVICE>.onrender.com/analyze -H "Content-Type: application/json" -d '{"text":"I love this movie!"}'

Troubleshooting

- 404s: confirm the Start Command / Procfile matches the app you expect (`src.day4:app` vs `src.day13:app`). The Procfile currently points to `day4`.
- OOM or build failures: use `requirements.deploy.txt` or the Dockerfile (no torch installed). If you need local inference, use a larger instance or Docker image that pre-bundles the model.
- HF API failures: ensure `HF_INFERENCE_API_TOKEN` is set and valid. Check Render logs for authorization errors (401/403) and timeout errors.

Automation and extras (I can add)

- `.render.yaml` to codify the service name, start command, and environment variables.
- GitHub Actions workflow to build and publish a Docker image and optionally trigger Render.
- Add a smoke-test Action that runs `curl /` after deploy and fails the pipeline if the wrong app is serving.

If you want, I can add any of the automation items above or switch the repo defaults so Render exposes `day13` instead of `day4` (or vice versa). Tell me which app you want exposed and whether you prefer Docker or Render's Python service, and I'll make the small edits or add the CI file.
