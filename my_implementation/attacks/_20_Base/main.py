# attacks/_00_BaseAttack/main.py
import os, json,sys, argparse, pandas as pd
from pathlib import Path
from tqdm import tqdm

from attacks._20_Base.base_attack import BaseAttack, BaseAttackConfig
from attacks.common.llm import LLM
from attacks.helpers import load_config, str2bool
from defense.defense_EA import DefenseEA

def run_base_attack(victim_llm_path, results_dir, dataset_path, what_ollama_model, api_ollama_vllm):
    here = Path(__file__).parent
    cfg_root = load_config(here / "configBase.yaml")["BaseAttack"]

    # model_path = cfg_root["victim_llm"]
    # data_csv   = cfg_root["data_path"]
    # out_dir    = cfg_root["output_dict"]
    temperature= cfg_root.get("temperature", 0.0)
    max_token  = cfg_root.get("max_token", 512)


    victim = LLM(victim_llm_path, temperature, max_token, api_ollama_vllm, what_ollama_model)
    attack_cfg = BaseAttackConfig()
    attacker = BaseAttack(attack_cfg)
    # defense  = DefenseEA() if use_defense else None

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

            # if defense:
                # messages[-1]["content"] = defense(messages[-1]["content"])

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