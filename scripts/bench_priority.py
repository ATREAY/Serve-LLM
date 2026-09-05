"""Phase 6: proves request.priority actually changes scheduling order under
real contention, against a live ServeLLM instance — not a unit test against
vLLM's scheduler in isolation.

Needs genuine queueing to show anything: if every concurrent request fits
under the engine's max_num_seqs immediately, nothing ever waits and priority
has nothing to act on. Point this at a model whose max_num_seqs is small
relative to --concurrency (see backend/router/models.yaml's comment on
"general" — the default deployment doesn't constrain it, so pass a small
max_num_seqs there and restart the gateway before running this, then revert).

Usage:
    python scripts/bench_priority.py --base-url http://dgx-v100-01:18742 \\
        --model general --concurrency 24 --max-tokens 200
"""

import argparse
import asyncio
import time

import httpx


async def _one_request(client: httpx.AsyncClient, model: str, priority: int, tag: str,
                        max_tokens: int, start: float) -> dict:
    t0 = time.perf_counter()
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": f"Write a short sentence about {tag}."}],
            "max_tokens": max_tokens,
            "priority": priority,
        },
    )
    resp.raise_for_status()
    return {
        "tag": tag,
        "priority": priority,
        "latency_s": time.perf_counter() - t0,
        "completed_at_s": time.perf_counter() - start,
    }


async def run(base_url: str, model: str, concurrency: int, max_tokens: int, timeout: float):
    half = concurrency // 2
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        start = time.perf_counter()
        tasks = [
            _one_request(client, model, priority=0, tag=f"high-{i}", max_tokens=max_tokens, start=start)
            for i in range(half)
        ] + [
            _one_request(client, model, priority=100, tag=f"low-{i}", max_tokens=max_tokens, start=start)
            for i in range(concurrency - half)
        ]
        results = await asyncio.gather(*tasks)

    results.sort(key=lambda r: r["completed_at_s"])
    for rank, r in enumerate(results):
        print(f"{rank:3d}  completed_at={r['completed_at_s']:6.2f}s  "
              f"latency={r['latency_s']:6.2f}s  priority={r['priority']:4d}  {r['tag']}")

    high = [r["latency_s"] for r in results if r["priority"] == 0]
    low = [r["latency_s"] for r in results if r["priority"] == 100]
    high_avg_rank = sum(rank for rank, r in enumerate(results) if r["priority"] == 0) / len(high)
    low_avg_rank = sum(rank for rank, r in enumerate(results) if r["priority"] == 100) / len(low)

    print()
    print(f"high-priority (0):   avg latency {sum(high)/len(high):6.2f}s   avg completion rank {high_avg_rank:5.1f}")
    print(f"low-priority  (100): avg latency {sum(low)/len(low):6.2f}s   avg completion rank {low_avg_rank:5.1f}")
    print()
    if high_avg_rank < low_avg_rank and sum(high) / len(high) < sum(low) / len(low):
        print("PRIORITY EFFECT CONFIRMED: high-priority requests finished earlier on both measures.")
    else:
        print("NO CLEAR PRIORITY EFFECT — likely no real queueing contention occurred "
              "(raise --concurrency, or lower the model's max_num_seqs and restart the gateway).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="general")
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    asyncio.run(run(args.base_url, args.model, args.concurrency, args.max_tokens, args.timeout))


if __name__ == "__main__":
    main()
