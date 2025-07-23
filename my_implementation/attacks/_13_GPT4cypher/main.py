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

def run_GPTcypher_attack(run_defense: bool = False) -> None:

    # ----------------– načtení YAML -------------------------------------
    yaml_path = os.path.join(os.path.dirname(__file__), "configCypher.yaml")
    cfg       = load_config(yaml_path)["GPT4Cypher"]

    # ----------------– victim LLM ---------------------------------------
    victim_llm = LLM(
        model_path  = cfg["victim_llm"],
        temperature = cfg.get("temperature", 0.0),
        max_tokens  = cfg.get("max_token", 512),
    )

    # ----------------– dataset -----------------------------------------
    data_path = cfg["data_path"]
    if data_path.lower().endswith(".csv"):
        df = pd.read_csv(data_path, index_col=0)
        if "Goal" not in df.columns:
            raise KeyError("Column 'Goal' not found in dataset")
        prompts = df["Goal"].dropna().tolist()
    elif data_path.lower().endswith(".txt"):
        with open(data_path, "r", encoding="utf-8") as fi:
            prompts = [line.strip() for line in fi if line.strip()]
    else:
        raise ValueError("data_path musí být .csv nebo .txt")

    # ----------------– výstup -------------------------------------------
    out_dir  = cfg["output_dict"]
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "_13_gpt4cipher.jsonl")

    # ----------------– konfigurace útočníka -----------------------------
    attacker_fields = {f.name for f in fields(GPT4CipherAttackerConfig)}
    attacker_kwargs = {k: v for k, v in cfg.items() if k in attacker_fields}
    attacker_cfg    = GPT4CipherAttackerConfig(**attacker_kwargs)
    attacker        = GPT4CipherAttacker(attacker_cfg)

    defense = DefenseEA() if run_defense else None

    # ----------------– hlavní smyčka ------------------------------------
    with open(out_file, "w", encoding="utf-8") as fo:
        for idx, prompt in tqdm(enumerate(prompts), total=len(prompts)):
            messages = [{"role": "user", "content": prompt}]
            messages = attacker.attack(messages)

            if defense:
                messages[-1]["content"] = defense(messages[-1]["content"])

            llm_resp = victim_llm.response(messages)

            json.dump(
                {
                    "id": idx,
                    "prompt_original": prompt,
                    "prompt_attack":  messages[-1]["content"],
                    "response":       llm_resp,
                },
                fo,
                ensure_ascii=False,
            )
            fo.write("\n")

    print(f"[INFO] Výsledky uloženy do {out_file}")
