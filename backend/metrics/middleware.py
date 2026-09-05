"""Prometheus metrics: HTTP-level (via middleware) and inference-level
(TTFT, tokens/sec, in-flight requests — updated directly by the gateway)."""

import time

from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

HTTP_REQUESTS = Counter(
    "servellm_http_requests_total", "HTTP requests", ["method", "path", "status"]
)
HTTP_LATENCY = Histogram(
    "servellm_http_request_latency_seconds", "HTTP request latency", ["method", "path"]
)

INFERENCE_REQUESTS = Counter(
    "servellm_inference_requests_total", "Inference requests", ["model", "endpoint", "status"]
)
INFLIGHT_REQUESTS = Gauge(
    "servellm_inflight_requests", "Requests currently being generated", ["model"]
)
TIME_TO_FIRST_TOKEN = Histogram(
    "servellm_ttft_seconds", "Time to first token", ["model"]
)
TOKENS_PER_SECOND = Histogram(
    "servellm_tokens_per_second", "Output tokens/sec for completed requests", ["model"]
)
PROMPT_TOKENS = Counter("servellm_prompt_tokens_total", "Prompt tokens processed", ["model"])
COMPLETION_TOKENS = Counter(
    "servellm_completion_tokens_total", "Completion tokens generated", ["model"]
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        path = request.url.path
        HTTP_REQUESTS.labels(request.method, path, response.status_code).inc()
        HTTP_LATENCY.labels(request.method, path).observe(elapsed)
        return response
