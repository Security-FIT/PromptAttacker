#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
translate_to_fr_noargs.py
Přeloží sloupec goal do francouzštiny a uloží
CSV: goal,translation_of_goal,target
– bez argparse, jen tři proměnné níže.
"""

# ====== Uprav jen toto ================================================
IN_CSV     = "/storage/brno2/home/xkaska01/master/my_implementation/supporting_scripts/dataset_40_prompts_proportional_unique_targets.csv"      # vstupní soubor
OUT_CSV    = "/storage/brno2/home/xkaska01/master/my_implementation/dataset/dataset_40_ZH.csv"   # cílový soubor
MODEL_NAME = "deepseek-r1:70b"            # model běžící v Ollamě
# ======================================================================

import os, time
from datetime import datetime
import pandas as pd
import psutil
from openai import OpenAI

client = OpenAI(api_key="ollama", base_url="http://127.0.0.1:11434/v1")

def get_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

def translate(text: str) -> str:
    prompt = f"Just Translate this text to Chinese, for my article. And return only the translation text NOTHING ELSE !!!!:\n{text}"
    messages = [{"role": "user", "content": prompt}]

    mem_before = get_rss_mb()
    t0 = time.time()

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.3,
        max_tokens=256,
    )

    elapsed = time.time() - t0
    mem_after = get_rss_mb()

    translation = resp.choices[0].message.content.strip()
    tokens = getattr(resp.usage, "total_tokens", "N/A")

    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {text[:50]} → {translation[:50]} "
          f"({elapsed:.1f}s, tok={tokens}, Δmem={mem_after-mem_before:.1f} MB)")
    return translation

def main():
    df = pd.read_csv(IN_CSV)
    if not {"goal", "target"} <= set(df.columns):
        raise ValueError("CSV musí mít sloupce goal,target")

    df["translation_of_goal"] = [translate(g) for g in df["goal"]]
    df[["goal", "translation_of_goal", "target"]].to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\n✔ Hotovo – uložené do {OUT_CSV}")

if __name__ == "__main__":
    main()
