# attacks/_00_BaseAttack/main.py
import os, json, argparse, pandas as pd
from pathlib import Path
from tqdm import tqdm

from attacks._20_Base.base_attack import BaseAttack, BaseAttackConfig
from attacks._20_Base.llm import LLM
from attacks.helpers import load_config
from defense.defense_EA import DefenseEA

def run_base_attack(use_defense: bool = False):
    here = Path(__file__).parent
    cfg_root = load_config(here / "configBase.yaml")["BaseAttack"]

    model_path = cfg_root["victim_llm"]
    data_csv   = cfg_root["data_path"]
    out_dir    = cfg_root["output_dict"]
    temperature= cfg_root.get("temperature", 0.0)
    max_token  = cfg_root.get("max_token", 512)


    victim = LLM(model_path, temperature, max_token)
    attack_cfg = BaseAttackConfig()
    attacker = BaseAttack(attack_cfg)
    defense  = DefenseEA() if use_defense else None

    df = pd.read_csv(data_csv)
    df.columns = [c.lower().strip() for c in df.columns]

    out_file = out_dir + "/" + "_20_base.json"
    with open(out_file, 'w', encoding='utf-8') as fo:
        fo.write("[\n")
        first = True

        for idx, row in enumerate(tqdm(df.itertuples(index=False),
                                       total=len(df),
                                       desc="BaseAttack")):
            original = str(row.goal)
            messages = [{"role":"user", "content": original}]

            messages = attacker.attack(messages)

            if defense:
                messages[-1]["content"] = defense(messages[-1]["content"])

            try:
                reply = victim.response(messages)
            except Exception as e:
                reply = f"[ERROR] {e}"

            entry = {
                "id":               idx,
                "original_prompt":  original,
                "attacked_prompt":  messages[-1]["content"],
                "response":         reply
            }

            if not first:
                fo.write(",\n")
            first = False

            fo.write(json.dumps(entry, ensure_ascii=False))
            fo.flush()

        fo.write("\n]\n")

    print(f"[INFO] Výstup uložen → {out_file}")