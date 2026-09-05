"""Phase 7: measures real throughput/latency for N concurrent requests against
a live ServeLLM instance — meant to be run twice against the same model under
two different backend/router/models.yaml max_num_seqs settings:

    max_num_seqs: 1    -> serialized baseline ("without batching")
    max_num_seqs unset -> vLLM default (256), continuous batching does its job

vLLM has no separate "disable continuous batching" toggle — the engine's
execution model IS continuous batching. max_num_seqs=1 is the honest way to
get a true serialized baseline out of the same engine/model/hardware, rather
than switching to a different framework (HF Transformers generate()) for the
baseline, which would also change kernel implementations, memory layout, etc.
and confound the comparison with more than just "batched vs not".

Usage:
    python scripts/bench_batching.py --base-url http://dgx-v100-01:18742 \\
        --model general --concurrency 16 --max-tokens 150
"""

import argparse
import asyncio
import time

import httpx


async def _one_request(client: httpx.AsyncClient, model: str, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Write a short paragraph about the ocean."}],
            "max_tokens": max_tokens,
        },
    )
    resp.raise_for_status()
    body = resp.json()
    return {
        "latency_s": time.perf_counter() - t0,
        "completion_tokens": body["usage"]["completion_tokens"],
    }


async def run(base_url: str, model: str, concurrency: int, max_tokens: int, timeout: float):
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        wall_start = time.perf_counter()
        results = await asyncio.gather(
            *[_one_request(client, model, max_tokens) for _ in range(concurrency)]
        )
        wall_elapsed = time.perf_counter() - wall_start

    latencies = [r["latency_s"] for r in results]
    total_tokens = sum(r["completion_tokens"] for r in results)

    print(f"concurrency:          {concurrency}")
    print(f"wall-clock time:      {wall_elapsed:.2f}s")
    print(f"avg per-request lat:  {sum(latencies)/len(latencies):.2f}s")
    print(f"min/max latency:      {min(latencies):.2f}s / {max(latencies):.2f}s")
    print(f"total completion tok: {total_tokens}")
    print(f"aggregate throughput: {total_tokens/wall_elapsed:.1f} tokens/s")
    print(f"requests/s:           {concurrency/wall_elapsed:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="general")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    asyncio.run(run(args.base_url, args.model, args.concurrency, args.max_tokens, args.timeout))


if __name__ == "__main__":
    main()
