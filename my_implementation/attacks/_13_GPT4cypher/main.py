## @file main.py
#  @brief Runner script for GPT4Cypher-style (Caesar cipher) prompt attack evaluation
#
#  This script runs the GPT4Cypher attack using parameters loaded from
#  `configCypher.yaml`. It loads a CSV dataset (expects a column named `goal`),
#  applies the attacker to transform each goal into a cipher-based adversarial
#  prompt, queries the victim LLM, deciphers the returned response, and stores
#  all results in JSON format.
#
#  @author Bc. Petr Kaška
#  @date 30.1.2026
#
#  Ownership / Contribution statement:
#   - This runner script was fully implemented by Bc. Petr Kaška.
#   - The orchestration logic, config loading, dataset handling, experiment loop,
#     and result serialization are original work by the author.
#   - The GPT4Cypher attacker implementation is imported from
#     `attacks/_13_GPT4cypher/attack_cypher.py` and integrated here into the
#     experimental framework.
#   - The victim model interface is provided via `attacks/common/llm.py` (LLM).

from __future__ import annotations

import argparse, sys
import json
import os
from dataclasses import fields

import pandas as pd
from tqdm import tqdm

from attacks._13_GPT4cypher.attack_cypher import (
    GPT4CipherAttacker,
    GPT4CipherAttackerConfig,
)
from attacks.common.helpers import load_config, str2bool
from attacks.common.llm import LLM  


def run_GPTcypher_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):

    yaml_path = os.path.join(os.path.dirname(__file__), "configCypher.yaml")
    cfg       = load_config(yaml_path)["GPT4Cypher"]

    victim_llm = LLM(
        model_path  = victim_llm_path,
        temperature = cfg.get("temperature", 0.0),
        max_tokens  = cfg.get("max_token", 512),
        ollama_model=what_ollama_model,
        use_ollama=api_ollama_vllm, 
    )

    df = pd.read_csv(dataset_path)

    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, "_13_gpt4cipher.json")

    attacker_fields = {f.name for f in fields(GPT4CipherAttackerConfig)}
    attacker_kwargs = {k: v for k, v in cfg.items() if k in attacker_fields}
    attacker_cfg    = GPT4CipherAttackerConfig(**attacker_kwargs)
    attacker        = GPT4CipherAttacker(attacker_cfg)

    entries = []
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

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 run_cypher.py victim_llm_path results_dir dataset_path api_ollama_vllm what_ollama_model")
        sys.exit(1)

    victim_llm_path = sys.argv[1]
    results_dir = sys.argv[2]
    dataset_path = sys.argv[3]
    api_ollama_vllm = str2bool(sys.argv[4].lower())
    what_ollama_model = sys.argv[5]

    run_GPTcypher_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)