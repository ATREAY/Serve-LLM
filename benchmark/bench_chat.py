"""Phase 12: general-purpose concurrent load generator against a live
ServeLLM instance's /v1/chat/completions. Reports latency percentiles
(p50/p95/p99), TTFT, and aggregate throughput — the reusable tool the
Phase 6-8 scripts (scripts/bench_priority.py, bench_batching.py,
bench_prefix_cache.py) are each narrower, purpose-built variants of.

Usage:
    python benchmark/bench_chat.py --base-url http://dgx-v100-01:18742 \\
        --model general --concurrency 16 --requests 32 --max-tokens 100
"""

import argparse
import asyncio
import json
import statistics
import time

import httpx


async def _one_request(client: httpx.AsyncClient, model: str, max_tokens: int, stream: bool) -> dict:
    t0 = time.perf_counter()
    if not stream:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Write two sentences about the weather."}],
                "max_tokens": max_tokens,
            },
        )
        resp.raise_for_status()
        body = resp.json()
        return {
            "latency_s": time.perf_counter() - t0,
            "ttft_s": None,
            "completion_tokens": body["usage"]["completion_tokens"],
        }

    ttft = None
    completion_tokens = 0
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Write two sentences about the weather."}],
            "max_tokens": max_tokens,
            "stream": True,
        },
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data: "):])
            content = chunk["choices"][0]["delta"].get("content")
            if content:
                completion_tokens += 1
                if ttft is None:
                    ttft = time.perf_counter() - t0
    return {"latency_s": time.perf_counter() - t0, "ttft_s": ttft, "completion_tokens": completion_tokens}


def _percentiles(values: list[float]) -> dict:
    s = sorted(values)
    n = len(s)
    return {
        "p50": s[int(n * 0.50)],
        "p95": s[min(int(n * 0.95), n - 1)],
        "p99": s[min(int(n * 0.99), n - 1)],
        "min": s[0],
        "max": s[-1],
        "avg": statistics.mean(s),
    }


async def run(base_url: str, model: str, concurrency: int, total_requests: int, max_tokens: int,
              stream: bool, timeout: float):
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(client):
        async with sem:
            return await _one_request(client, model, max_tokens, stream)

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        wall_start = time.perf_counter()
        results = await asyncio.gather(*[_bounded(client) for _ in range(total_requests)])
        wall_elapsed = time.perf_counter() - wall_start

    latencies = [r["latency_s"] for r in results]
    lat_pcts = _percentiles(latencies)
    total_tokens = sum(r["completion_tokens"] for r in results)

    print(f"requests:             {total_requests} (concurrency {concurrency})")
    print(f"wall-clock time:      {wall_elapsed:.2f}s")
    print(f"latency p50/p95/p99:  {lat_pcts['p50']:.2f}s / {lat_pcts['p95']:.2f}s / {lat_pcts['p99']:.2f}s")
    print(f"latency min/avg/max:  {lat_pcts['min']:.2f}s / {lat_pcts['avg']:.2f}s / {lat_pcts['max']:.2f}s")
    if stream:
        ttfts = [r["ttft_s"] for r in results if r["ttft_s"] is not None]
        if ttfts:
            ttft_pcts = _percentiles(ttfts)
            print(f"ttft p50/p95:         {ttft_pcts['p50']:.3f}s / {ttft_pcts['p95']:.3f}s")
    print(f"total completion tok: {total_tokens}")
    print(f"aggregate throughput: {total_tokens/wall_elapsed:.1f} tokens/s")
    print(f"requests/s:           {total_requests/wall_elapsed:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="general")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--stream", action="store_true", help="measure TTFT via SSE streaming")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    asyncio.run(run(args.base_url, args.model, args.concurrency, args.requests,
                     args.max_tokens, args.stream, args.timeout))


if __name__ == "__main__":
    main()
