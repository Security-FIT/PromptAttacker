#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline překlad 'goal' -> čínština (Simplified) přes Hugging Face Transformers.
Bez argumentů; cesty a chování nastavíš v konstantách níže.
Model: Helsinki-NLP/opus-mt-en-zh (cca 300 MB).
"""

import sys
from typing import Dict, List
import pandas as pd

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False

# ========= KONFIGURACE =========
IN_CSV  = "/storage/brno2/home/xkaska01/master/my_implementation/dataset/cysecbench_adv_small_ZH.csv"
OUT_CSV = "/storage/brno2/home/xkaska01/master/my_implementation/dataset/test.csv"

MODEL_NAME = "Helsinki-NLP/opus-mt-en-zh"  # EN -> ZH (Simplified)
BATCH_SIZE = 16
OVERWRITE  = True   # True = přepíše i již vyplněné překlady
MAX_LENGTH = 512     # max délka sekvence pro model
NUM_BEAMS  = 4       # beam search pro lepší kvalitu
# ===============================


def init_translator():
    device = 0 if torch.cuda.is_available() else -1
    # Explicitní načtení (kvůli robustnosti)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    trans = pipeline(
        "translation",
        model=mdl,
        tokenizer=tok,
        device=device
    )
    return trans


def batched(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def translate_unique_texts(translator, texts: List[str]) -> Dict[str, str]:
    """Přeloží unikátní texty po dávkách a vrátí mapu {origin: zh}."""
    mapping: Dict[str, str] = {}
    todo = [t for t in texts if t and t.strip()]
    if not todo:
        return mapping

    iterable = batched(todo, BATCH_SIZE)
    if TQDM:
        iterable = tqdm(list(iterable), total=(len(todo) + BATCH_SIZE - 1) // BATCH_SIZE, desc="Translating (offline)")

    for batch in iterable:
        try:
            outputs = translator(
                batch,
                max_length=MAX_LENGTH,
                num_beams=NUM_BEAMS,
                batch_size=BATCH_SIZE
            )
        except RuntimeError as e:
            # typicky OOM na GPU -> zkusíme menší batch
            if "out of memory" in str(e).lower() and len(batch) > 1:
                half = max(1, len(batch)//2)
                for sub in batched(batch, half):
                    sub_out = translator(sub, max_length=MAX_LENGTH, num_beams=NUM_BEAMS, batch_size=max(1, half))
                    for src, o in zip(sub, sub_out):
                        mapping[src] = o["translation_text"]
                continue
            else:
                raise

        for src, o in zip(batch, outputs):
            mapping[src] = o["translation_text"]

    return mapping


def main():
    df = pd.read_csv(IN_CSV, dtype=str, keep_default_na=False)

    # Ověření sloupců
    required = {"goal", "target"}
    if not required.issubset(df.columns):
        raise ValueError(f"Vstupní CSV musí obsahovat sloupce: {sorted(required)}. Nalezené: {list(df.columns)}")

    if "translation_of_goal" not in df.columns:
        df["translation_of_goal"] = ""

    translator = init_translator()

    # Vybereme řádky k překladu
    if OVERWRITE:
        idxs = df.index.tolist()
    else:
        idxs = [i for i in df.index if not (df.at[i, "translation_of_goal"] or "").strip()]

    # Připravíme unikátní texty (deduplikace kvůli rychlosti)
    goals = [(df.at[i, "goal"] or "").strip() for i in idxs]
    unique_goals = sorted(set([g for g in goals if g]))

    mapping = translate_unique_texts(translator, unique_goals)

    # Zápis zpět do dataframe
    for i in idxs:
        g = (df.at[i, "goal"] or "").strip()
        df.at[i, "translation_of_goal"] = mapping.get(g, df.at[i, "translation_of_goal"])

    # Zachovat pořadí sloupců
    ordered = ["goal", "translation_of_goal", "target"]
    ordered_existing = [c for c in ordered if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered_existing]
    df = df[ordered_existing + remaining]

    df.to_csv(OUT_CSV, index=False)
    print(f"✅ Hotovo: {len(df)} řádků uloženo do: {OUT_CSV}")
    if not OVERWRITE:
        print(f"   (doplněno {len(idxs)} překladů; už vyplněné zůstaly beze změny)")
    else:
        print("   (všechny překlady byly přepsány)")


if __name__ == "__main__":
    main()
