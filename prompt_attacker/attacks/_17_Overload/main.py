## @file main.py
#  @brief Runner script for prompt Overload-based jailbreak attack
#
#  This script evaluates an Overload-style jailbreak attack in which the
#  original user prompt is transformed into an overloaded prompt containing
#  excessive instructions or contextual noise. The objective is to overwhelm
#  the victim model’s safety mechanisms by increasing prompt complexity and
#  instruction density.
#
#  The script:
#   - Loads attack and model parameters from `configOverload.yaml`
#   - Reads a CSV dataset containing adversarial goals (column: `goal`)
#   - Applies the Overload transformation via `OverloadAttacker`
#   - Queries the victim LLM with the overloaded prompt
#   - Stores original prompts, transformed prompts, and model responses
#     in JSON format
#
#  @author Bc. Petr Kaška
#  @date 1.2.2026
#
#  Ownership / Contribution statement:
#   - This runner script was fully implemented by Bc. Petr Kaška.
#   - The experiment orchestration, configuration handling, attacker
#     instantiation, LLM querying, and result serialization are original
#     work by the author.
#   - The Overload attack module is integrated as part of the author’s
#     experimental evaluation framework.
#

import os
import json, sys
import pandas as pd
from tqdm import tqdm
from dataclasses import fields 

from attacks._17_Overload.overload_attack import OverloadAttacker, OverloadAttackerConfig
from attacks.common.helpers import load_config, str2bool
from attacks.common.llm import LLM


def run_overload_attack(victim_llm_path, results_dir, dataset_path,  api_ollama_vllm, what_ollama_model):

    cfg = load_config(os.path.join(os.path.dirname(__file__), "configOverload.yaml"))
    cfg_ol = cfg["Overload"]

    victim_llm = LLM(
        model_path=victim_llm_path,
        temperature=cfg_ol.get("temperature", 0.0),
        max_tokens=cfg_ol.get("max_token", 512),
        ollama_model=what_ollama_model,
        use_ollama=api_ollama_vllm, 
    )

    df = pd.read_csv(dataset_path)

    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, "_17_overload.json")  # JSON Lines
    attacker_field_names = {f.name for f in fields(OverloadAttackerConfig)}

    attacker_kwargs = {k: v for k, v in cfg_ol.items() if k in attacker_field_names}
    attacker_cfg = OverloadAttackerConfig(**attacker_kwargs)
    attacker     = OverloadAttacker(attacker_cfg)
    entries = []

    with open(out_file, "w", encoding="utf-8") as fo:
        for idx, prompt in tqdm(enumerate(df["goal"])):
            attack_messages = attacker.attack([{"role": "user", "content": prompt}])

            llm_resp = victim_llm.response(attack_messages)

            entry = {
                "id": idx,
                "original_prompt": prompt,
                "prompt": attack_messages[-1]["content"],
                "response": llm_resp,
            }
            entries.append(entry)
        fo.write(json.dumps(entries, ensure_ascii=False) + "\n")
        fo.flush()

    print(f"[INFO] Results saved to {out_file}")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 run_cypher.py victim_llm_path results_dir dataset_path api_ollama_vllm what_ollama_model")
        sys.exit(1)

    victim_llm_path = sys.argv[1]
    results_dir = sys.argv[2]
    dataset_path = sys.argv[3]
    api_ollama_vllm = str2bool(sys.argv[4].lower())
    what_ollama_model = sys.argv[5]

    run_overload_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)