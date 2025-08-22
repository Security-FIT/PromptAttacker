# attacks/_4_ReNeLLM/main.py
import json, argparse, pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

from attacks._22_Renellm.renellm_attack import ReNeLLMAttack, ReNeLLMConfig
from attacks.common.llm import LLM
from attacks.helpers import load_config
from defense.defense_EA import DefenseEA

def run_renellm_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(script_dir + "/configRenellm.yaml")["ReNeLLM"]


    # ---------- config hodnoty -----------------------------------------
    # model_path = cfg["victim_llm"]
    # data_csv   = cfg["data_path"]
    out_dir    = Path(results_dir); out_dir.mkdir(parents=True, exist_ok=True)
    params     = ReNeLLMConfig(iter_max=cfg.get("iter_max", 20),
                               use_cot=cfg.get("cot", False))

    attack_llm = LLM(
                victim_llm_path,
                cfg.get("temperature", 0.0),
                cfg.get("max_token", 512),
                ollama_model= "qwen2.5:7b",
                use_ollama=api_ollama_vllm) # TADY POZOR NECHAVAM ZATIM PORAD NA FALSE PROTOZE MUSIM ZMENIT KOD UVNITR!!!!!! PROTOZE TAM POUZIVAM NADSTAVBU PRO VLLM A NE OLAMU

    victim_llm = LLM(
                victim_llm_path,
                cfg.get("temperature", 0.0),
                cfg.get("max_token", 512),
                ollama_model=what_ollama_model,
                use_ollama=api_ollama_vllm)
    
    # stejný gen-config použijeme pro rewrite i pro inference
    gen_cfg = dict(max_n_tokens=cfg.get("max_token", 512),
                   temperature=cfg.get("temperature", 0.0),
                   logprobs=False, seed=None)

    attack = ReNeLLMAttack(params,
                            rewrite_llm=attack_llm,
                            rewrite_gen_cfg=gen_cfg,
                            api_ollama_vllm=api_ollama_vllm,
                            what_ollama_model=what_ollama_model)
    # defense = DefenseEA() if use_defense else None

    df = pd.read_csv(dataset_path)
    df.columns = [c.lower() for c in df.columns]


    out_file = out_dir / "_22_renellm.json"
    with out_file.open("w", encoding="utf-8") as fo:
        for idx, row in enumerate(tqdm(df.itertuples(index=False),
                                       total=len(df),
                                       desc="ReNeLLM")):
            harmful = str(row.goal)
            # Vygenerovat útoky pro všechny metody
            # attack = ReNeLLMAttack(params,
            #                        rewrite_llm=llm.client,
            #                        rewrite_gen_cfg=gen_cfg,
            #                        api_ollama_vllm=api_ollama_vllm,
            #                        what_ollama_model=what_ollama_model)
            for method, log, messages in attack.generate(harmful):
                # volitelná obrana
                # if defense:
                #     messages[-1]["content"] = defense(messages[-1]["content"])

                # poslat prompt do modelu
                try:
                    response = victim_llm.response(messages)
                except Exception as e:
                    response = f"[ERROR] {e}"

                # zapsat JSONL
                fo.write(json.dumps({
                    "id":               idx,
                    "original_prompt":  harmful,
                    "prompt":     messages[-1]["content"],
                    "response":         response
                }, ensure_ascii=False) + "\n")
                fo.flush()

    print(f"[INFO] Výstup uložen → {out_file}")