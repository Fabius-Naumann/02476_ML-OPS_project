import time

from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Histogram, generate_latest

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

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


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
