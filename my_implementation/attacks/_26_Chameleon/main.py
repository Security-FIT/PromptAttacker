#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py – jednoduchý spouštěč „CodeChameleon“ útoku s pevnou šifrou binary_tree.
Rozhraní = úplně stejné jako run_flip_attack().
"""

import os, sys
import json
import pandas as pd
from tqdm import tqdm

# interní moduly projektu
from attacks.common.llm import LLM
from attacks.helpers import load_config, str2bool
from attacks._26_Chameleon.encrypt import get_encrypted_query
from attacks._26_Chameleon.template import get_prompts_code, get_prompts_text

# --------------------------------------------------------------------------- #
# Hlavní funkce – stejná signatura jako run_flip_attack
# --------------------------------------------------------------------------- #
def run_chameleon_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):
    """
    Spustí Binary‑Tree jail­break útok (bez evaluace).

    Parametry – totožné s run_flip_attack:
    ----------------------------------------------------------
    victim_llm_path  : str   –  cesta / jméno LLM modelu
    results_dir      : str   –  složka pro výstupní soubory
    dataset_path     : str   –  CSV se sloupkem 'problem' (nebo 'goal')
    api_ollama_vllm  : bool  –  True ⇒ použít Ollama/VLLM backend
    what_ollama_model: str   –  název Ollama modelu
    """
    # ------------------------------------------------------------------ #
    # 1) Konfigurace – načtení YAML *stejnou* cestou jako run_flip_attack
    # ------------------------------------------------------------------ #
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configChameleon.yaml")  # klidně přejmenuj
    cfg         = load_config(config_path)
    cfgCham     = cfg["Chameleon"]          # očekáváme sekci "Chameleon" v YAML

    print("[CONFIG]", cfgCham)

    temperature   = cfgCham.get("temperature", 1.0)
    max_token     = cfgCham.get("max_token",  512)
    begin         = cfgCham.get("begin", 0)
    end           = cfgCham.get("end", None)              # None → do konce

    # ------------------------------------------------------------------ #
    # 2) Inicializace klienta LLM (identicky jako ve FlipAttack skriptu)
    # ------------------------------------------------------------------ #
    victim_llm = LLM(
        model_path=victim_llm_path,
        temperature=temperature,
        max_tokens=max_token,
        ollama_model=what_ollama_model,
        use_ollama=api_ollama_vllm,
    )
    print("[INFO] LLM inicializován")

    # ------------------------------------------------------------------ #
    # 3) Načtení datasetu
    # ------------------------------------------------------------------ #
    df = pd.read_csv(dataset_path)
    if "problem" in df.columns:
        prompts_series = df["problem"]
    elif "goal" in df.columns:
        prompts_series = df["goal"]
    else:
        raise ValueError("CSV musí obsahovat sloupec 'problem' nebo 'goal'.")

    if end is None or end > len(prompts_series):
        end = len(prompts_series)

    # ------------------------------------------------------------------ #
    # 4) Výstupní soubor
    # ------------------------------------------------------------------ #
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, "_26_Chameleon.json")

    # ------------------------------------------------------------------ #
    # 5) Hlavní smyčka: encrypt → prompt → LLM → záznam
    # ------------------------------------------------------------------ #
    entries = []
    with open(output_file, "w", encoding="utf-8") as fo:
        for idx, original_problem in tqdm(
            enumerate(prompts_series[begin:end]), total=end - begin
        ):
            print(f"[INFO] → #{idx}: {original_problem[:60]}…")

            # 5.1 zašifruj dotaz (binary_tree)
            encrypted_problem = get_encrypted_query(original_problem, "binary_tree")

            # 5.2 postav prompt podle požadovaného stylu
            # prompt = get_prompts_code([encrypted_problem], "binary_tree")[0]

            prompt = get_prompts_text([encrypted_problem], "binary_tree")[0]

            # 5.3 zavolej LLM
            llm_response = victim_llm.response([{"role": "user", "content": prompt}])

            # 5.4 ulož výsledek
            entry = {
                "id": idx,
                # "encrypt_rule": "binary_tree",
                # "prompt_style": prompt_style,
                "original_prompt": original_problem,
                "prompt": prompt,
                "response": llm_response,
            }
            entries.append(entry)
        fo.write(json.dumps(entries, ensure_ascii=False) + "\n")
        fo.flush()

    print(f"[INFO] Hotovo – uložené výsledky: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 run_cypher.py victim_llm_path results_dir dataset_path api_ollama_vllm what_ollama_model")
        sys.exit(1)

    victim_llm_path = sys.argv[1]
    results_dir = sys.argv[2]
    dataset_path = sys.argv[3]
    api_ollama_vllm = str2bool(sys.argv[4].lower())
    what_ollama_model = sys.argv[5]

    run_chameleon_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)