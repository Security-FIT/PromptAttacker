# TENTO FILE JE MUUUUUUUUUJ

import os
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import yaml
import sys # Importujeme sys pro ukončení programu

from defense.defense_EA import DefenseEA
from attacks._11_Pair.llm import LLM
from attacks._11_Pair.pair_attack import PairAttack, PairAttackConfig


def run_pair_attack(run_defense: bool = False):
    defense = DefenseEA() if run_defense else None
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    cfg_path    = os.path.join(script_dir, "configPair.yaml")

    cfg_yaml = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    cfgP     = cfg_yaml["Pair"]

    # ------------------------------------------------------------------------------------------------
    attack_cfg = PairAttackConfig(
        attacker_llm_path=cfgP["attacker_llm_path"],
        target_llm_path=cfgP["target_llm_path"],
        target_str=cfgP.get("target_str", "Sure,"),
        n_iterations=cfgP.get("n_iterations", 5),
        temperature=cfgP.get("temperature", 1.0),
        max_tokens=cfgP.get("max_tokens", 1024),
        early_stop_threshold=cfgP.get("early_stop_threshold", 8),
    )
    attacker = PairAttack(attack_cfg)

    # ------------------------------------------------------------------------------------------------
    data = pd.read_csv(cfgP["data_path"])
    
    found_goal_col = [col for col in data.columns if col == "Goal"]
    goal_col = found_goal_col[0] 

    begin, end = cfgP.get("begin", 0), cfgP.get("end", len(data))

    out_dir = Path(cfgP["output_dict"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "_11_pair.json"

    with open(out_file, "w", encoding="utf-8") as fo:
        for idx, goal in tqdm(enumerate(data[goal_col][begin:end]), total=end - begin):
            result = attacker.generate(goal)

            prompt_adv = result["adversarial_prompt"]
            if run_defense:
                prompt_adv = defense(prompt_adv)

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

