#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import time

from llm import LLM


###########################################
#           HARD-CODED SETTINGS
###########################################
INPUT_DIR = "/storage/brno2/home/xkaska01/master/my_implementation/dataset/my_experimentdataset"
OUTPUT_DIR = "my_generated_dataset_testgemma"

MODEL = "/storage/brno2/home/xkaska01/master/my_implementation/models/gemma"   # nebo co používáš jako model_path
USE_OLLAMA = False
MAX_TOKENS = 500
TEMPERATURE = 0.8
TIMEOUT = 1200

BATCH_SIZE = 16   # velikost batch – klidně zkus vyšší/nižší podle VRAM
###########################################


def process_all_files():
    # start celkového měření
    total_start = time.time()

    # inicializace modelu – přizpůsob podle signatury tvého LLM
    llm = LLM(
        model_path=MODEL,          # jestli používáš vLLM, dej sem cestu k modelu
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        ollama_model="llama3.1:8b",
        use_ollama=USE_OLLAMA,
        timeout=TIMEOUT,
    )

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    for file in input_dir.iterdir():
        if file.suffix != ".json":
            continue

        file_start = time.time()
        print(f"\n=== Zpracovávám soubor: {file.name} ===")

        # načtení původního JSONu
        with file.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  [ERROR] Neplatný JSON ({file.name}): {e}")
                continue

        if not isinstance(data, list):
            print(f"  [WARN] {file.name} neobsahuje list objektů, přeskočeno.")
            continue

        total_items = len(data)
        print(f"  Počet záznamů: {total_items}")

        # buffers pro batch
        batch_prompts = []
        batch_indices = []

        # projdeme všechny položky a posíláme je po batchech
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue

            prompt = item.get("prompt")
            if not isinstance(prompt, str):
                print(f"  [WARN] položka {idx} nemá string 'prompt', přeskočeno.")
                continue

            batch_prompts.append(prompt)
            batch_indices.append(idx)

            # když naplníme batch, pošleme ho do modelu
            if len(batch_prompts) >= BATCH_SIZE:
                print(f"    → batch {batch_indices[0]}–{batch_indices[-1]} / {total_items}",
                      end="\r", flush=True)

                responses = llm.response_batch(batch_prompts)

                # přiřadíme odpovědi zpátky do dat
                for i, resp in zip(batch_indices, responses):
                    data[i]["response"] = resp

                # vyčistit batch
                batch_prompts = []
                batch_indices = []

        # poslední neúplný batch (pokud něco zbylo)
        if batch_prompts:
            print(f"    → batch {batch_indices[0]}–{batch_indices[-1]} / {total_items}",
                  end="\r", flush=True)

            responses = llm.response_batch(batch_prompts)
            for i, resp in zip(batch_indices, responses):
                data[i]["response"] = resp

        # uložení do výstupní složky (stejné jméno souboru)
        out_path = output_dir / file.name
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        file_elapsed = time.time() - file_start
        print(f"\n✓ Hotovo: {file.name} za {file_elapsed:.2f} s")

    total_elapsed = time.time() - total_start
    print("\n=== ZPRACOVÁNÍ DOKONČENO ===")
    print(f"Celkový čas: {total_elapsed:.2f} s (~{total_elapsed/60:.2f} min)")


if __name__ == "__main__":
    process_all_files()
