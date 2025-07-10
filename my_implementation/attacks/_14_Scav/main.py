# encoding: utf-8
"""
main_scav.py — Runner for SCAV attack
=====================================
• Vybere správné CSV podle configu „8b_or_70b“  
• Z CSV bere sloupec *original_instruction* jako eval prompty  
• Volitelně spustí EA defense (--defense)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields

import pandas as pd
from tqdm import tqdm

# Lokální importy
from attacks._14_Scav.attack_scav import ScavAttacker, ScavAttackerConfig
from attacks.helpers import load_config
from attacks._14_Scav.llm import LLM
from defense.defense_EA import DefenseEA


# --------------------------------------------------------------------------- #
#                               Pomocné funkce                                #
# --------------------------------------------------------------------------- #
def load_prompts(csv_path: str) -> list[str]:
    """Vrátí seznam originálních promptů z CSV (první kolona)."""
    df = pd.read_csv(csv_path, header=None, usecols=[0])
    return df.dropna().iloc[:, 0].tolist()


# --------------------------------------------------------------------------- #
#                                 Hlavní běh                                  #
# --------------------------------------------------------------------------- #
def run_scav_attack(run_defense: bool = False) -> None:
    yaml_path = os.path.join(os.path.dirname(__file__), "configScav.yaml")
    cfg       = load_config(yaml_path)["Scav"]

    # ------------------ vyber datasetu podle klíče ------------------------ #
    model_size = cfg.get("8b_or_70b", "8b").lower()
    if model_size not in ("8b", "70b"):
        raise ValueError("8b_or_70b musí být \"8b\" nebo \"70b\"")

    dataset_key = f"optimized_dataset_{model_size}"
    if dataset_key not in cfg:
        raise KeyError(f"V YAML chybí '{dataset_key}'")
    dataset_csv = cfg[dataset_key]

    prompts = load_prompts(dataset_csv)

    # ------------------------- inicializace LLM --------------------------- #
    victim_llm = LLM(
        model_path = cfg["victim_llm"],
        temperature = cfg.get("temperature", 0.0),
        max_tokens = cfg.get("max_token", 512),
    )

    # --------------------------- výstupní soubor -------------------------- #
    out_dir  = cfg["output_dict"]
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"_14_scav{model_size}.jsonl")

    # --------------------------- útočník ---------------------------------- #
    attacker_fields = {f.name for f in fields(ScavAttackerConfig)}
    attacker_kwargs = {k: v for k, v in cfg.items() if k in attacker_fields}
    attacker_kwargs["model_size"] = model_size
    attacker = ScavAttacker(ScavAttackerConfig(**attacker_kwargs))

    defense = DefenseEA() if run_defense else None

    # ----------------------------- hlavní smyčka -------------------------- #
    with open(out_file, "w", encoding="utf-8") as fo:
        for idx, prompt in tqdm(enumerate(prompts), total=len(prompts)):
            messages = [{"role": "user", "content": prompt}]
            messages = attacker.attack(messages)

            if defense:
                messages[-1]["content"] = defense(messages[-1]["content"])

            llm_resp = victim_llm.response(messages[-1]["content"])

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
