# attacks/_4_ReNeLLM/main.py
import json, argparse, pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

from attacks._22_Renellm.renellm_attack import ReNeLLMAttack, ReNeLLMConfig
from attacks.common.llm import LLM
from attacks.helpers import load_config
from defense.defense_EA import DefenseEA

def run_renellm_attack(use_defense=False):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(script_dir + "/configRenellm.yaml")["ReNeLLM"]


    # ---------- config hodnoty -----------------------------------------
    model_path = cfg["victim_llm"]
    data_csv   = cfg["data_path"]
    out_dir    = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    params     = ReNeLLMConfig(iter_max=cfg.get("iter_max", 20),
                               use_cot=cfg.get("cot", False))

    llm     = LLM(model_path,
                  cfg.get("temperature", 0.0),
                  cfg.get("max_token", 512))

    # stejný gen-config použijeme pro rewrite i pro inference
    gen_cfg = dict(max_n_tokens=cfg.get("max_token", 512),
                   temperature=cfg.get("temperature", 0.0),
                   logprobs=False, seed=None)

    attack  = ReNeLLMAttack(params, rewrite_llm=llm.client,
                            rewrite_gen_cfg=gen_cfg)
    defense = DefenseEA() if use_defense else None

    df = pd.read_csv(data_csv)
    df.columns = [c.lower() for c in df.columns]


    out_file = out_dir / "_22_renellm.json"
    with out_file.open("w", encoding="utf-8") as fo:
        for idx, row in enumerate(tqdm(df.itertuples(index=False),
                                       total=len(df),
                                       desc="ReNeLLM")):
            harmful = str(row.goal)
            # Vygenerovat útoky pro všechny metody
            attack = ReNeLLMAttack(params,
                                   rewrite_llm=llm.client,
                                   rewrite_gen_cfg=gen_cfg)
            for method, log, messages in attack.generate_all(harmful):
                # volitelná obrana
                if defense:
                    messages[-1]["content"] = defense(messages[-1]["content"])

                # poslat prompt do modelu
                try:
                    response = llm.response(messages)
                except Exception as e:
                    response = f"[ERROR] {e}"

                # zapsat JSONL
                fo.write(json.dumps({
                    "id":               idx,
                    "method":           method,
                    "category":         getattr(row, "category", ""),
                    "original_prompt":  harmful,
                    "log":              log,
                    "final_prompt":     messages[-1]["content"],
                    "response":         response
                }, ensure_ascii=False) + "\n")
                fo.flush()

    print(f"[INFO] Výstup uložen → {out_file}")