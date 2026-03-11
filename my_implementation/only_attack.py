#!/usr/bin/env python3
# TENTO FILE JE muj 

import sys
import os
import json
from tqdm import tqdm
from pathlib import Path
import time

from attacks.common.llm import LLM
from attacks.common.helpers import str2bool


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
    # 1) Soubor neexistuje
    if not os.path.exists(path):
        print(f"[WARN] Soubor neexistuje – vytvářím prázdný JSON: {path}")
        write_json(path, [])
        return []

    # 2) Soubor existuje, ale je prázdný
    if os.path.getsize(path) == 0:
        print(f"[WARN] Soubor je prázdný – inicializuji jako []: {path}")
        write_json(path, [])
        return []

    # 3) Soubor existuje, ale není validní JSON
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
    # podpora různých klíčů, které se v tvých datasetech objevují
    return (
        d.get("original_prompt")
    )


def pick_prompt(d: dict):
    # ber primárně 'prompt', jinak další rozumné aliasy
    return (
        d.get("prompt")
    )


def process_file(in_file: str, out_dir: str, victim_llm_path: str, use_ollama: bool, ollama_model: str):
    """
    Načte JSON (pole objektů), pro každý objekt pošle `prompt` do LLM a uloží
    JSON výsledků ve formátu:
      [{id, original_prompt, prompt, response}, ...]
    """
    data = read_json(in_file)
    if not isinstance(data, list):
        print(f"[WARN] {in_file} není JSON pole – přeskočeno")
        return

    # Inicializace LLM (jednou pro soubor)
    llm = LLM(
        model_path=victim_llm_path,
        temperature=0.0,
        max_tokens=512,
        ollama_model=ollama_model,
        use_ollama=use_ollama,
    )

    out = []
    for item in tqdm(data, desc=f"{Path(in_file).name}", leave=False):
        _id = item.get("id", item.get("idx"))
        original = pick_original(item)
        prompt = pick_prompt(item)

        if not prompt:
            resp = "<<NO PROMPT FOUND>>"
        else:
            try:
                resp = llm.response(prompt)
                time.sleep(0.1)
            except Exception as e:
                resp = f"<<LLM ERROR: {e}>>"

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
    print(f"[OK] {Path(in_file).name} -> {out_path}")


def main():
    """
    Usage (stejný styl jako tvůj run_cypher):
      python3 only_attack.py victim_llm_path input_dir output_dir api_ollama_vllm what_ollama_model
    """
    if len(sys.argv) != 6:
        print("Usage: python3 only_attack.py victim_llm_path input_dir output_dir api_ollama_vllm what_ollama_model")
        sys.exit(1)

    victim_llm_path = sys.argv[1]

    dataset_path = sys.argv[2]
    results_dir = sys.argv[3]

    use_ollama      = str2bool(sys.argv[4].lower())
    ollama_model    = sys.argv[5]

    # if not os.path.isdir(dataset_path):
    #     print(f"[ERR] dataset_path není složka: {dataset_path}")
    #     sys.exit(2)

    # ensure_dir(dataset_path)

    # # Vem všechny .json v kořeni složky (podsložku 'jobs' ignoruj)
    # files = [
    #     os.path.join(dataset_path, f)
    #     for f in sorted(os.listdir(dataset_path))
    #     if f.lower().endswith(".json") and f != "jobs"
    # ]

    # if not files:
    #     print(f"[WARN] Ve složce {dataset_path} nejsou žádné .json soubory")
    #     sys.exit(0)

    print(f"[INFO] Vstupní složka: {dataset_path}")
    print(f"[INFO] Výstupní složka: {results_dir}")
    print(f"[INFO] Model (ollama): {ollama_model}")
    print(f"[INFO] use_ollama:     {use_ollama}")
    # print(f"[INFO] Soubory:        {len(files)}\n")

    process_file(
        in_file=dataset_path,
        out_dir=results_dir,
        victim_llm_path=victim_llm_path,
        use_ollama=use_ollama,
        ollama_model=ollama_model,
    )

    print("\n[DONE] Všechny soubory zpracovány.")


if __name__ == "__main__":
    main()
