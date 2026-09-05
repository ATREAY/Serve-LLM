"""Phase 8: measures whether vLLM's automatic prefix caching actually reduces
TTFT for requests sharing a long common prompt prefix (e.g. a system prompt),
against a live ServeLLM instance.

Requests are fired SEQUENTIALLY, not concurrently — deliberately, to isolate
the prefix-cache effect from continuous batching (Phase 7). If fired
concurrently, any speedup could be batching, not cache reuse; run one at a
time, and any drop in TTFT from request 2 onward can only be the shared
prefix's KV blocks being reused rather than recomputed.

Meant to be run twice, same two-config pattern as Phases 6-7: once against a
model with enable_prefix_caching: false (baseline — TTFT should stay flat
across requests) and once with it: true (TTFT should drop sharply after the
first request warms the cache).

Usage:
    python scripts/bench_prefix_cache.py --base-url http://dgx-v100-01:18742 \\
        --model general --requests 8
"""

import argparse
import asyncio
import json
import time

import httpx

# ~500 tokens of shared instructional prefix, deliberately long relative to
# the short per-request suffix below — makes the prefix-cache saving (or lack
# of one) dominate the measured TTFT rather than being lost in noise.
SHARED_PREFIX = (
    "You are a careful, precise research assistant helping a graduate student "
    "review technical literature. When answering, be concise, cite specific "
    "reasoning, and avoid speculation beyond what is stated. Consider the "
    "following context before responding: this student is working on a "
    "systems research project involving large language model inference "
    "serving, GPU memory management, request scheduling policies, and "
    "quantitative benchmarking methodology. They value precise, falsifiable "
    "claims over vague generalities, and they will follow up with clarifying "
    "questions if an answer is incomplete or ambiguous. Keep responses under "
    "three sentences unless explicitly asked for more detail. Do not repeat "
    "the question back before answering. Do not include unnecessary caveats "
    "or disclaimers. Focus only on what was actually asked. "
)

QUESTIONS = [
    "What is a KV cache?",
    "What is PagedAttention?",
    "What is continuous batching?",
    "What is speculative decoding?",
    "What is a LoRA adapter?",
    "What is quantization?",
    "What is prefix caching?",
    "What is chunked prefill?",
]


async def _one_streamed_request(client: httpx.AsyncClient, model: str, question: str) -> dict:
    t0 = time.perf_counter()
    ttft = None
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "system", "content": SHARED_PREFIX},
                         {"role": "user", "content": question}],
            "max_tokens": 20,
            "stream": True,
        },
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]" or ttft is not None:
                continue
            # The gateway's first SSE event is always an empty role-delta
            # chunk emitted before generation starts (see
            # backend/gateway/main.py's _stream_chat) — near-zero latency
            # regardless of prefix caching. Skip past it to the first chunk
            # that actually carries generated text; that's the real TTFT.
            chunk = json.loads(line[len("data: "):])
            if chunk["choices"][0]["delta"].get("content"):
                ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return {"question": question, "ttft_s": ttft, "total_s": total}


async def run(base_url: str, model: str, n: int, timeout: float):
    questions = (QUESTIONS * ((n // len(QUESTIONS)) + 1))[:n]
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        results = []
        for i, q in enumerate(questions):
            r = await _one_streamed_request(client, model, q)
            print(f"request {i}: ttft={r['ttft_s']:.3f}s  total={r['total_s']:.3f}s  ({r['question']})")
            results.append(r)

    first = results[0]
    rest = results[1:]
    print()
    print(f"request 0 (cold):        ttft={first['ttft_s']:.3f}s")
    print(f"requests 1..{len(rest)} (avg): ttft={sum(r['ttft_s'] for r in rest)/len(rest):.3f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="general")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    asyncio.run(run(args.base_url, args.model, args.requests, args.timeout))


if __name__ == "__main__":
    main()
