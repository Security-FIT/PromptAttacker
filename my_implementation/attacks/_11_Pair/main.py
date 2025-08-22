# TENTO FILE JE MUUUUUUUUUJ

import os
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import yaml
import sys # Importujeme sys pro ukončení programu

from defense.defense_EA import DefenseEA
from attacks._11_Pair.pair_attack import PairAttack, PairAttackConfig


def run_pair_attack(victim_llm_path, results_dir, dataset_path, what_ollama_model, api_ollama_vllm):
    # defense = DefenseEA() if run_defense else None
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    cfg_path    = os.path.join(script_dir, "configPair.yaml")

    cfg_yaml = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    cfgP     = cfg_yaml["Pair"]

    # ------------------------------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------------------------------
    adv_bench = pd.read_csv(dataset_path)


    
    out_dir = Path(results_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "_11_pair.json"
    print("ZDEEE")
    with open(out_file, "w", encoding="utf-8") as fo:
        for idx, goal in tqdm(enumerate(adv_bench["goal"])):
            result = attacker.generate(goal)

            prompt_adv = result["adversarial_prompt"]
            # if run_defense:
                # prompt_adv = defense(prompt_adv)

            entry = {
                "id": idx,
                "goal": goal,
                "prompt": prompt_adv,
                "response": result["model_response"],
                "score": result["score"],
            }
            fo.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fo.flush()

    print(f"[INFO] Výsledky uloženy do {out_file}")

