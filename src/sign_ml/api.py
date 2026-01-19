import json
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Histogram, generate_latest

# --------------------
# Prometheus metrics (M28)
# --------------------

REQUEST_COUNT = Counter(
    "request_count_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "http_status"],
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Latency of HTTP requests in seconds",
    ["endpoint"],
)

# --------------------
# App
# --------------------

app = FastAPI()

# --------------------
# Inference logging (M27)
# --------------------

LOG_PATH = Path("data/inference_logs/predictions.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_prediction(input_summary: dict, output_summary: dict) -> None:
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_summary,
        "output": output_summary,
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


# --------------------
# Basic endpoints
# --------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(dummy_input: float = 0.0):
    """
    Dummy prediction endpoint for M27.
    Logs input-output pairs for drift analysis.
    """

    # Dummy "model"
    prediction = 1
    confidence = 0.85

    log_prediction(
        input_summary={"dummy_input": dummy_input},
        output_summary={"class": prediction, "confidence": confidence},
    )

    return {"class": prediction, "confidence": confidence}


# --------------------
# Metrics middleware (M28)
# --------------------

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    try:
        response: Response = await call_next(request)
    except Exception:
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            http_status="500",
        ).inc()
        raise

    latency = time.time() - start_time
    endpoint = request.url.path

    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        http_status=response.status_code,
    ).inc()

    return response


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
