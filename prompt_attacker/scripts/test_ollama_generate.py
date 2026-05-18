#!/usr/bin/env python3
## @file test_ollama_generate.py
#  @brief Minimal Ollama `/api/generate` smoke test.
#
#  This helper verifies that a running Ollama server can load a selected model
#  and produce a short response within the configured timeout. It is useful for
#  checking GPU allocations and model-loading behavior before longer runs.
#
#  @author Bc. Petr Kaska
#  @date 1.2.2026
#
#  Ownership / Contribution statement:
#   - This file was designed and implemented by Bc. Petr Kaska.
#   - The environment-driven smoke-test workflow and request handling are original
#     project utilities.
#   - The implementation uses standard Python HTTP utilities and Ollama API usage.

import json
import os
import time
import urllib.error
import urllib.request


HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.getenv("OLLAMA_MODEL", "llama2:7b")
PROMPT = os.getenv("OLLAMA_PROMPT", "Reply with exactly one short sentence: Ollama works.")
TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "60"))
NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "32"))


def post_json(path, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        HOST + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "stream": False,
        "options": {
            "num_predict": NUM_PREDICT,
            "temperature": 0,
        },
    }
    print(f"[OLLAMA-TEST] host={HOST}", flush=True)
    print(f"[OLLAMA-TEST] model={MODEL}", flush=True)
    print(f"[OLLAMA-TEST] num_predict={NUM_PREDICT}, timeout={TIMEOUT}s", flush=True)
    start = time.time()
    try:
        data = post_json("/api/generate", payload, TIMEOUT)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"[OLLAMA-TEST][HTTP {e.code}] {body}")
    elapsed = time.time() - start
    response = str(data.get("response", "")).strip()
    print(f"[OLLAMA-TEST] elapsed={elapsed:.2f}s", flush=True)
    print(f"[OLLAMA-TEST] response={response!r}", flush=True)
    if not response:
        raise SystemExit("[OLLAMA-TEST][FAIL] Empty response")
    print("[OLLAMA-TEST][OK] Ollama generation works", flush=True)


if __name__ == "__main__":
    main()
