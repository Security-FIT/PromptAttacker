## @file main.py
#  @brief Runner script for PAIR (Prompt Attack via Iterative Refinement) attack
#
#  This script executes the PAIR attack, an iterative prompt refinement
#  strategy where adversarial prompts are progressively optimized based
#  on model feedback and an internal scoring function.
#
#  The script:
#   - loads PAIR-specific configuration from a YAML file,
#   - initializes PairAttack and PairAttackConfig,
#   - iterates over a dataset of adversarial goals,
#   - generates optimized adversarial prompts,
#   - queries the victim LLM,
#   - stores prompts, responses, and attack scores in JSON format.
#
#  @author Bc. Petr Kaška
#  @date 30.1.2026
#

import datetime
import os
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import yaml
import sys 
from attacks.common.helpers import str2bool

from attacks._11_Pair.pair_attack import PairAttack, PairAttackConfig

def run_pair_attack(victim_llm_path, results_dir, dataset_path, what_ollama_model, api_ollama_vllm):
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    cfg_path    = os.path.join(script_dir, "configPair.yaml")

    cfg_yaml = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    cfgP     = cfg_yaml["Pair"]

    attack_cfg = PairAttackConfig(
        attacker_llm_path=victim_llm_path,
        target_llm_path=victim_llm_path,
        target_str=cfgP.get("target_str", "Sure,"),
        n_iterations=cfgP.get("n_iterations", 5),
        temperature=cfgP.get("temperature", 1.0),
        max_tokens=cfgP.get("max_tokens", 1024),
        early_stop_threshold=cfgP.get("early_stop_threshold", 8),
        api_ollama_vllm=api_ollama_vllm,
        what_ollama_model=what_ollama_model,
    )
    attacker = PairAttack(attack_cfg)

    adv_bench = pd.read_csv(dataset_path)

    entries = []
    out_dir = Path(results_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"_11_pair.json"
    with open(out_file, "w", encoding="utf-8") as fo:
        for idx, goal in tqdm(enumerate(adv_bench["goal"])):
            result = attacker.generate(goal)

            prompt_adv = result["adversarial_prompt"]

            entry = {
                "id": idx,
                "original_prompt": goal,
                "prompt": prompt_adv,
                "response": result["model_response"],
                "score": result["score"],
            }
            entries.append(entry)
        fo.write(json.dumps(entries, ensure_ascii=False) + "\n")
        fo.flush()

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

    run_pair_attack(victim_llm_path, results_dir, dataset_path, what_ollama_model, api_ollama_vllm)