from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Request, Response
from prometheus_client import Counter, Histogram, generate_latest

# --------------------
# Paths (M27)
# --------------------

LOG_PATH = Path("data/inference_logs/predictions.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

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
# Inference logging (M27)
# --------------------


def log_prediction(input_summary: dict, output_summary: dict) -> None:
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "input": input_summary,
        "output": output_summary,
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


# --------------------
# Metrics middleware (M28)
# --------------------


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


def metrics_response() -> Response:
    return Response(generate_latest(), media_type="text/plain")
