# encoding: utf-8
"""
main_gpt4cipher.py — Runner for GPT‑4 Cipher attack
==================================================
Spouštěcí skript po vzoru *main_scav.py*.  Vybere dataset z `data_path`, na
každý prompt aplikuje *GPT4CipherAttacker* a výsledek předá do victim LLM.
Volitelně lze zapnout EA defence pomocí `--defense`.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields

import pandas as pd
from tqdm import tqdm

# Lokální importy
from attacks._13_GPT4cypher.attack_cypher import (
    GPT4CipherAttacker,
    GPT4CipherAttackerConfig,
)
from attacks.helpers import load_config
from attacks.common.llm import LLM  # sdílíme jednoduchý wrapper
from defense.defense_EA import DefenseEA


# ---------------------------------------------------------------------------
#                               Runner
# ---------------------------------------------------------------------------

def run_GPTcypher_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):

    # ----------------– načtení YAML -------------------------------------
    yaml_path = os.path.join(os.path.dirname(__file__), "configCypher.yaml")
    cfg       = load_config(yaml_path)["GPT4Cypher"]

    # ----------------– victim LLM ---------------------------------------
    victim_llm = LLM(
        model_path  = victim_llm_path,
        temperature = cfg.get("temperature", 0.0),
        max_tokens  = cfg.get("max_token", 512),
        ollama_model=what_ollama_model,
        use_ollama=api_ollama_vllm, 
    )

    # ----------------– dataset -----------------------------------------
    # data_path = cfg["data_path"]
    df = pd.read_csv(dataset_path)

    # ----------------– výstup -------------------------------------------
    
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, "_13_gpt4cipher.json")

    # ----------------– konfigurace útočníka -----------------------------
    attacker_fields = {f.name for f in fields(GPT4CipherAttackerConfig)}
    attacker_kwargs = {k: v for k, v in cfg.items() if k in attacker_fields}
    attacker_cfg    = GPT4CipherAttackerConfig(**attacker_kwargs)
    attacker        = GPT4CipherAttacker(attacker_cfg)

    # defense = DefenseEA() if run_defense else None
    entries = []
    # ----------------– hlavní smyčka ------------------------------------
    with open(out_file, "w", encoding="utf-8") as fo:
        for idx, prompt in tqdm(
                enumerate(df["goal"])):
            messages = [{"role": "user", "content": prompt}]
            messages = attacker.attack(messages)


            llm_resp = victim_llm.response(messages)
            deciphered_answer = attacker._caesar_decipher(llm_resp)  
            entry = {
                "id": idx,
                "prompt_original": prompt,
                "prompt":  messages[-1]["content"],
                "response":       deciphered_answer,
            }
            entries.append(entry)


        fo.write(json.dumps(entries, ensure_ascii=False) + "\n")

    print(f"[INFO] Výsledky uloženy do {out_file}")
