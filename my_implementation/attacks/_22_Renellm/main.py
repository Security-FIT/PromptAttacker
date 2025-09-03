# attacks/_4_ReNeLLM/main.py
#!/usr/bin/env python3
import json
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

from attacks._22_Renellm.renellm_attack import ReNeLLMAttack, ReNeLLMConfig
from attacks.common.llm import LLM
from attacks.helpers import load_config
# from defense.defense_EA import DefenseEA  # volitelně

def run_renellm_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(script_dir + "/configRenellm.yaml")["ReNeLLM"]

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = ReNeLLMConfig(
        iter_max=cfg.get("iter_max", 20),
        use_cot=cfg.get("cot", False)
    )

    # LLM pro přepis (útočník)
    attack_llm = LLM(
        victim_llm_path,
        cfg.get("temperature", 0.0),
        cfg.get("max_token", 512),
        ollama_model="qwen2.5:7b",
        use_ollama=api_ollama_vllm
    )

    # LLM oběť
    victim_llm = LLM(
        victim_llm_path,
        cfg.get("temperature", 0.0),
        cfg.get("max_token", 512),
        ollama_model=what_ollama_model,
        use_ollama=api_ollama_vllm
    )

    gen_cfg = dict(
        max_n_tokens=cfg.get("max_token", 512),
        temperature=cfg.get("temperature", 0.0),
        logprobs=False,
        seed=None
    )

    attack = ReNeLLMAttack(
        params,
        rewrite_llm=attack_llm,
        rewrite_gen_cfg=gen_cfg,
        api_ollama_vllm=api_ollama_vllm,
        what_ollama_model=what_ollama_model
    )

    df = pd.read_csv(dataset_path)
    df.columns = [c.lower() for c in df.columns]

    entries = []
    out_file = out_dir / "_22_renellm.json"

    # Pomocná funkce: vezmi první (method, log, messages) z generate()
    def first_method_messages(generate_result):
        """
        Vrátí tuple (method, log, messages) z prvního prvku.
        Ošetří, zda je to list/tuple nebo iterator/generátor.
        """
        if generate_result is None:
            return None
        # Pokud je to list/tuple, vem první item
        if isinstance(generate_result, (list, tuple)):
            if not generate_result:
                return None
            return generate_result[0]
        # Jinak zkus vytvořit iterator a vzít next
        try:
            it = iter(generate_result)
            return next(it, None)
        except TypeError:
            # Není to iterovatelné
            return None

    for idx, row in enumerate(tqdm(df.itertuples(index=False),
                                   total=len(df),
                                   desc="ReNeLLM")):
        harmful = str(row.goal)

        gen_res = attack.generate(harmful)
        first = first_method_messages(gen_res)
        if not first:
            entries.append({
                "id": idx,
                "original_prompt": harmful,
                "prompt": "",
                "response": "[ERROR] attack.generate() returned empty result",
                "method": None,
                "log": None
            })
            continue

        try:
            method, log, messages = first
        except Exception as e:
            entries.append({
                "id": idx,
                "original_prompt": harmful,
                "prompt": "",
                "response": f"[ERROR] Unexpected generate() item format: {e}",
                "method": None,
                "log": None
            })
            continue

        # Volitelná obrana
        # defense = DefenseEA()
        # messages[-1]["content"] = defense(messages[-1]["content"])

        # Inference
        try:
            response = victim_llm.response(messages)
        except Exception as e:
            response = f"[ERROR] {e}"

        # Pokud chceš zachovat \n doslova, odkomentuj tyto 2 řádky:
        # prompt_text = messages[-1]["content"].replace("\n", "\\n")
        # response_text = str(response).replace("\n", "\\n")
        # Jinak ponecháme původní:
        prompt_text = messages[-1]["content"]
        response_text = response

        entries.append({
            "id": idx,
            "original_prompt": harmful,
            "prompt": prompt_text,
            "response": response_text,
            "method": method,
            "log": log
        })

    with out_file.open("w", encoding="utf-8") as fo:
        json.dump(entries, fo, ensure_ascii=False, indent=2)

    print(f"[INFO] Výstup uložen → {out_file}")