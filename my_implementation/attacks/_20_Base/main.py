## @file main.py
#  @brief Runner script for the BaseAttack prefix/suffix baseline
#
#  This script evaluates a minimal prefix/suffix-based prompt manipulation
#  attack (BaseAttack). The attack performs no semantic transformation and
#  simply prepends and/or appends fixed strings to the last user message.
#
#  The script:
#   - Loads configuration parameters from `configBase.yaml`
#   - Reads a CSV dataset containing a `goal` column
#   - Applies the BaseAttack to each prompt
#   - Queries the victim LLM
#   - Stores original prompts, modified prompts, and model responses in JSON
#
#  @author Bc. Petr Kaška
#  @date 1.2.2026
#
#  Ownership / Contribution statement:
#   - This runner script was fully implemented by Bc. Petr Kaška.
#   - The experiment orchestration, dataset handling, LLM interaction,
#     baseline attack execution, and result serialization are original work
#     by the author.
#

import os, json,sys, argparse, pandas as pd
from pathlib import Path
from tqdm import tqdm

from attacks._20_Base.base_attack import BaseAttack, BaseAttackConfig
from attacks.common.llm import LLM
from attacks.common.helpers import load_config, str2bool

def run_base_attack(victim_llm_path, results_dir, dataset_path, what_ollama_model, api_ollama_vllm):
    here = Path(__file__).parent
    cfg_root = load_config(here / "configBase.yaml")["BaseAttack"]

    temperature= cfg_root.get("temperature", 0.0)
    max_token  = cfg_root.get("max_token", 512)

    victim = LLM(victim_llm_path, temperature, max_token, api_ollama_vllm, what_ollama_model)
    attack_cfg = BaseAttackConfig()
    attacker = BaseAttack(attack_cfg)

    df = pd.read_csv(dataset_path)
    df.columns = [c.lower().strip() for c in df.columns]
    entries = []
    out_file = results_dir + "/" + "_20_base.json"
    with open(out_file, 'w', encoding='utf-8') as fo:

        for idx, row in enumerate(tqdm(df.itertuples(index=False),
                                       total=len(df),
                                       desc="BaseAttack")):
            original = str(row.goal)
            messages = [{"role":"user", "content": original}]

            messages = attacker.attack(messages)

            try:
                reply = victim.response(messages)
            except Exception as e:
                reply = f"[ERROR] {e}"

            entry = {
                "id":               idx,
                "original_prompt":  original,
                "prompt":  messages[-1]["content"],
                "response":         reply
            }

            entries.append(entry)

        fo.write(json.dumps(entries, ensure_ascii=False))
        fo.flush()


    print(f"[INFO] Výstup uložen → {out_file}")



if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 run_cypher.py victim_llm_path results_dir dataset_path api_ollama_vllm what_ollama_model")
        sys.exit(1)

    victim_llm_path = sys.argv[1]
    results_dir = sys.argv[2]
    dataset_path = sys.argv[3]
    api_ollama_vllm = str2bool(sys.argv[4].lower())
    what_ollama_model = sys.argv[5]

    run_base_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)