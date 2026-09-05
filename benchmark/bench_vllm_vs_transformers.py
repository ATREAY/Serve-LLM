"""Phase 12: the comparison the original architecture doc specifically asked
for — vLLM vs plain HF `transformers.generate()` — which Phase 7 didn't cover
(that compared vLLM against itself with continuous batching on/off, not
against a different framework).

Requires a GPU (loads TinyLlama directly via `transformers`, separately from
whatever ServeLLM itself has loaded) — run via sbatch/srun, not on the login
node. Both sides run sequentially, one request at a time: this isolates
per-framework generation overhead, not batching (see scripts/bench_batching.py
for that axis) — an apples-to-apples single-stream comparison.

Usage (from a GPU node/job):
    python benchmark/bench_vllm_vs_transformers.py \\
        --base-url http://dgx-v100-01:18742 --model general --requests 5 --max-tokens 100
"""

import argparse
import time

import httpx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "Write two sentences about the ocean."


def bench_transformers(n: int, max_tokens: int) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
    model.eval()

    messages = [{"role": "user", "content": PROMPT}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

    results = []
    for _ in range(n):
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        completion_tokens = output.shape[1] - inputs["input_ids"].shape[1]
        results.append({"latency_s": elapsed, "completion_tokens": completion_tokens})
    return results


def bench_vllm(base_url: str, model: str, n: int, max_tokens: int) -> list[dict]:
    results = []
    with httpx.Client(base_url=base_url, timeout=120) as client:
        for _ in range(n):
            t0 = time.perf_counter()
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": PROMPT}],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                },
            )
            resp.raise_for_status()
            elapsed = time.perf_counter() - t0
            results.append({
                "latency_s": elapsed,
                "completion_tokens": resp.json()["usage"]["completion_tokens"],
            })
    return results


def _summarize(name: str, results: list[dict]) -> None:
    total_latency = sum(r["latency_s"] for r in results)
    total_tokens = sum(r["completion_tokens"] for r in results)
    avg_latency = total_latency / len(results)
    print(f"{name}:")
    print(f"  requests:       {len(results)}")
    print(f"  avg latency:    {avg_latency:.2f}s")
    print(f"  total tokens:   {total_tokens}")
    print(f"  throughput:     {total_tokens/total_latency:.1f} tokens/s (sum over sequential requests)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="ServeLLM gateway, e.g. http://dgx-v100-01:18742")
    parser.add_argument("--model", default="general")
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    print("=== plain HF transformers.generate() (naive, no batching, no PagedAttention) ===")
    t_results = bench_transformers(args.requests, args.max_tokens)
    _summarize("transformers", t_results)

    print("\n=== vLLM via ServeLLM gateway ===")
    v_results = bench_vllm(args.base_url, args.model, args.requests, args.max_tokens)
    _summarize("vllm", v_results)

    t_avg = sum(r["latency_s"] for r in t_results) / len(t_results)
    v_avg = sum(r["latency_s"] for r in v_results) / len(v_results)
    print(f"\nvLLM avg latency is {t_avg / v_avg:.2f}x faster than plain transformers.generate()"
          if v_avg < t_avg else
          f"\nplain transformers.generate() avg latency is {v_avg / t_avg:.2f}x faster than vLLM"
          f" (unexpected — investigate before trusting this number)")


if __name__ == "__main__":
    main()
