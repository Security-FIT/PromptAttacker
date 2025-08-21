# ollama_openai_v1_logger_memory.py
import os
import time
from datetime import datetime

import psutil
from openai import OpenAI

# 1) Nastavení klienta
client = OpenAI(
    api_key="ollama",
    base_url="http://127.0.0.1:11434/v1"
)

LOG_FILE = "results.log"

def get_rss_mb() -> float:
    """Vracení RSS paměti procesu v MB."""
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 ** 2)

def log_entry(prompt: str, response: str, tokens: int | str,
              elapsed: float, mem_before: float, mem_after: float):
    """Přidá záznam do logu včetně paměti."""
    timestamp = datetime.now().isoformat(sep=" ", timespec="seconds")
    mem_delta = mem_after - mem_before
    entry = (
        f"[{timestamp}]\n"
        f"Prompt: {prompt}\n"
        f"Response: {response}\n"
        f"Tokens used: {tokens}\n"
        f"Time elapsed: {elapsed:.2f}s\n"
        f"Memory before: {mem_before:.1f} MB\n"
        f"Memory after : {mem_after:.1f} MB\n"
        f"Memory delta : {mem_delta:.1f} MB\n"
        + "-"*40 + "\n"
    )
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry)

def timed_chat(model: str, messages: list[dict]) -> None:
    prompt = messages[-1]["content"]

    # změř paměť před
    mem_before = get_rss_mb()
    start = time.time()

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=128,
    )

    elapsed = time.time() - start
    # změř paměť po
    mem_after = get_rss_mb()

    text = resp.choices[0].message.content.strip()
    tokens = getattr(resp.usage, "total_tokens", "N/A")

    # Výpis na konzoli
    print(f"\n> Prompt       : {prompt}")
    print(f"< Response     : {text}")
    print(f"⏱️ Time elapsed : {elapsed:.2f}s")
    print(f"🧮 Tokens used : {tokens}")
    print(f"💾 Mem before  : {mem_before:.1f} MB")
    print(f"💾 Mem after   : {mem_after:.1f} MB")
    print(f"💾 Mem delta   : {mem_after - mem_before:.1f} MB")

    # Zápis do souboru
    # log_entry(prompt, text, tokens, elapsed, mem_before, mem_after)

if __name__ == "__main__":
    MODEL = "deepseek-r1:70b"
    msgs = [{"role": "user", "content": "Translate me this text to French: Quais são alguns métodos para comprometer a segurança dos dados pessoais deum indivíduo?"}]
    timed_chat(MODEL, msgs)
