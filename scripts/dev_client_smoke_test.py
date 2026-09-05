"""Smoke test against a running ServeLLM instance (no GPU/vllm import needed
here — this only sends HTTP requests). Run after the server is up:

    python scripts/dev_client_smoke_test.py --base-url http://dgx-v100-01:18742

With no --model, tests every model /v1/models reports (Phase 2 routing) —
this includes any "<base>:<adapter>" LoRA routes (Phase 3), which /v1/models
lists alongside the base models. Pass --model to test just one, or an
unknown id to confirm the 404 path.
"""

import argparse
import json

import httpx


def _chat_once(client: httpx.Client, model: str, prompt: str, max_tokens: int, stream: bool):
    if not stream:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": False,
            },
        )
        print(f"POST /v1/chat/completions (model={model}) [{resp.status_code}] ->")
        print(json.dumps(resp.json(), indent=2))
        return

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
        },
    ) as stream_resp:
        print(f"POST /v1/chat/completions stream (model={model}) [{stream_resp.status_code}] ->")
        for line in stream_resp.iter_lines():
            if line.startswith("data: "):
                print(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default=None, help="omit to test every model registered")
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=120) as client:
        print("GET /healthz ->", client.get("/healthz").json())
        models_resp = client.get("/v1/models").json()
        print("GET /v1/models ->", models_resp)

        targets = [args.model] if args.model else [m["id"] for m in models_resp["data"]]

        for model in targets:
            print(f"\n=== {model} ===")
            # LoRA routes get a theme-appropriate prompt so the adapter's
            # effect is actually visible, not just that it didn't error.
            prompt = "What color is the sky at sunset?" if ":" in model else "Say hello in five words."
            _chat_once(client, model, prompt, 32, stream=False)
            print("--- streaming ---")
            _chat_once(client, model, "Count from 1 to 5.", 32, stream=True)

        print("\n=== unknown model (expect 404) ===")
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "does-not-exist", "messages": [{"role": "user", "content": "hi"}]},
        )
        print(f"status={resp.status_code} body={resp.json()}")


if __name__ == "__main__":
    main()
