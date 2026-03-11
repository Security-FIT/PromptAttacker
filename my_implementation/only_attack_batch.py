#!/usr/bin/env python3
# TENTO FILE JE BATCH VARIANTA only_attack.py

import sys
import os
import json
from tqdm import tqdm
from pathlib import Path
import time

from attacks.common.llm import LLM
from attacks.common.helpers import str2bool


BATCH_SIZE = 4


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def read_json(path: str):
    """
    Načte JSON soubor.
    Pokud:
    - soubor neexistuje
    - je prázdný
    - má nevalidní JSON
    → vytvoří validní prázdný JSON [] a vrátí []
    """
    if not os.path.exists(path):
        print(f"[WARN] Soubor neexistuje – vytvářím prázdný JSON: {path}")
        write_json(path, [])
        return []

    if os.path.getsize(path) == 0:
        print(f"[WARN] Soubor je prázdný – inicializuji jako []: {path}")
        write_json(path, [])
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Soubor není validní JSON: {path}\n       → {e}\n       Inicializuji jako []")
        write_json(path, [])
        return []


def write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def pick_original(d: dict):
    return d.get("original_prompt")


def pick_prompt(d: dict):
    return d.get("prompt")


def batched_llm_calls(llm: LLM, prompts: list[str]) -> list[str]:
    """
    Wrapper: pokud má LLM metodu response_batch, použije se.
    Jinak fallback na sekvenční volání response().
    """
    if not prompts:
        return []

    # Má LLM batch podporu?
    if hasattr(llm, "response_batch") and callable(getattr(llm, "response_batch")):
        try:
            return llm.response_batch(prompts)
        except Exception as e:
            print(f"[WARN] Chyba při batch inference, fallback na sekvenční režim: {e}")

    # Fallback – sekvenční (pomalejší)
    out = []
    for p in prompts:
        try:
            out.append(llm.response(p))
        except Exception as e:
            out.append(f"<<LLM ERROR (single): {e}>>")
    return out


def process_file(in_file: str, out_dir: str, victim_llm_path: str, use_ollama: bool, ollama_model: str):
    """
    Načte JSON (pole objektů), pro každý objekt pošle `prompt` do LLM v batchech
    a uloží JSON výsledků ve formátu:
      [{id, original_prompt, prompt, response}, ...]
    """
    data = read_json(in_file)
    if not isinstance(data, list):
        print(f"[WARN] {in_file} není JSON pole – přeskočeno")
        return

    # Inicializace LLM
    llm = LLM(
        model_path=victim_llm_path,
        temperature=0.8,
        max_tokens=512,
        ollama_model=ollama_model,
        use_ollama=use_ollama,
    )

    # Připravíme si indexy a prompty, které opravdu pošleme do modelu
    indices = []
    prompts = []
    meta = []  # (id, original_prompt, prompt)

    for idx, item in enumerate(data):
        _id = item.get("id", item.get("idx"))
        original = pick_original(item)
        prompt = pick_prompt(item)

        meta.append((_id, original, prompt))

        if not prompt:
            # necháme to zatím, vyřešíme při rekonstrukci
            continue

        indices.append(idx)
        prompts.append(prompt)

    print(f"[BATCH] Počet záznamů celkem: {len(data)}, z toho s promptem: {len(prompts)}")
    all_responses = [""] * len(data)

    # Batche přes prompty
    for start in tqdm(range(0, len(prompts), BATCH_SIZE), desc=f"{Path(in_file).name} [batch]", leave=False):
        end = min(start + BATCH_SIZE, len(prompts))
        batch_prompts = prompts[start:end]
        batch_indices = indices[start:end]

        batch_resps = batched_llm_calls(llm, batch_prompts)

        # namapujeme odpovědi zpět na správné indexy v původním poli
        for idx_in_data, resp in zip(batch_indices, batch_resps):
            all_responses[idx_in_data] = resp

        # případná pauza mezi batchi (když bys řešil nějaké rate-limity)
        time.sleep(0.01)

    # Poskládáme výstup
    out = []
    for i, (item_meta) in enumerate(meta):
        _id, original, prompt = item_meta

        if not prompt:
            resp = "<<NO PROMPT FOUND>>"
        else:
            raw_resp = all_responses[i]
            if raw_resp == "":
                resp = "<<NO RESPONSE>>"
            else:
                resp = raw_resp

        out.append(
            {
                "id": _id,
                "original_prompt": original,
                "prompt": prompt,
                "response": resp,
            }
        )

    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, os.path.basename(in_file))
    write_json(out_path, out)
    print(f"[OK-BATCH] {Path(in_file).name} -> {out_path}")


def main():
    """
    Usage:
      python3 only_attack_batch.py victim_llm_path input_json output_dir api_ollama_vllm what_ollama_model
    (stejná signatura jako only_attack.py)
    """
    if len(sys.argv) != 6:
        print("Usage: python3 only_attack_batch.py victim_llm_path input_json output_dir api_ollama_vllm what_ollama_model")
        sys.exit(1)

    victim_llm_path = sys.argv[1]
    dataset_path = sys.argv[2]   # tady je přímo cesta k JSON souboru
    results_dir = sys.argv[3]

    use_ollama = str2bool(sys.argv[4].lower())
    ollama_model = sys.argv[5]

    print(f"[INFO-BATCH] Vstupní soubor: {dataset_path}")
    print(f"[INFO-BATCH] Výstupní složka: {results_dir}")
    print(f"[INFO-BATCH] Model (ollama/vLLM): {ollama_model}")
    print(f"[INFO-BATCH] use_ollama:         {use_ollama}")
    print(f"[INFO-BATCH] BATCH_SIZE:         {BATCH_SIZE}")

    process_file(
        in_file=dataset_path,
        out_dir=results_dir,
        victim_llm_path=victim_llm_path,
        use_ollama=use_ollama,
        ollama_model=ollama_model,
    )

    print("\n[DONE-BATCH] Soubor zpracován.")


if __name__ == "__main__":
    main()
