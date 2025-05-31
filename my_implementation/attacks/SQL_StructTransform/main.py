import os, json, yaml, pandas as pd
from tqdm import tqdm
import gc
import torch

from attacks.SQL_StructTransform.llm import LLM
from attacks.SQL_StructTransform.SQL_attack import SQLAttack
from defense.defense_EA import DefenseEA 
from attacks.helpers import load_config 


# ---------- PIPELINE -------------------------------------------
def run_sql_attack(run_defense: bool = False):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configStruct.yaml")
    cfg = load_config(config_path)
    cfg = cfg["SQL"]

    # print(cfg)
    # exit(0)

    # LLM instance – M_attack
    attacker = LLM(
        model_path  = cfg['attacker_llm'],
        temperature = cfg['temperature_attack'],
        max_tokens  = cfg['max_tokens'],
    )
    # LLM instance – M_target
    target = LLM(
        model_path  = cfg['target_llm'],
        temperature = cfg['temperature_target'],
        max_tokens  = cfg['max_tokens'],
    )
    sql_attack = SQLAttack(
        attacker_llm = attacker,
        few_shot     = cfg['few_shot'],
        num_attempts = cfg['num_attempts']
    )
    defense = DefenseEA()

    df = pd.read_csv(cfg["data_path"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    out_path = os.path.join(cfg["output_dir"], "direct_sql.jsonl")

    with open(out_path, "w", encoding="utf-8") as fo:
        for idx, goal in tqdm(
                enumerate(df["query"][cfg["begin"]:cfg["end"]]),
                total=cfg["end"]-cfg["begin"]):
            log, prompts = sql_attack.generate(goal)

            if run_defense:
                prompts[-1]["content"] = defense(prompts[-1]["content"])

            response = target.response(prompts)

            fo.write(json.dumps({
                "id": idx,
                "goal": goal,
                "sql_prompt": prompts[-1]["content"],
                "response": response
            }, ensure_ascii=False) + "\n")
            fo.flush()

    print(f"[INFO] DirectSQL finished ➜ {out_path}")
