## @file run_renellm_attack.py
#  @brief Runner script for the ReNeLLM iterative rewriting jailbreak attack
#
#  This script evaluates the ReNeLLM attack, which performs iterative prompt
#  rewriting using a dedicated rewriting language model in order to bypass
#  safety and alignment constraints of a victim LLM.
#
#  @author Bc. Petr Kaška
#  @date 1.2.2026
#
#  Ownership / Contribution statement:
#   - This runner script was fully designed and implemented by Bc. Petr Kaška.
#   - The experiment control flow, dataset handling, error handling,
#     multi-method result parsing, and result serialization are original work
#     by the author.
#   - The script integrates the ReNeLLMAttack module into the unified attack
#     evaluation framework without reusing external runner implementations.
#

import json, sys
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

from attacks._22_Renellm.renellm_attack import ReNeLLMAttack, ReNeLLMConfig
from attacks.common.llm import LLM
from attacks.common.helpers import load_config, str2bool

def run_renellm_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(script_dir + "/configRenellm.yaml")["ReNeLLM"]

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = ReNeLLMConfig(
        iter_max=cfg.get("iter_max", 20),
        use_cot=cfg.get("cot", False)
    )

    attack_llm = LLM(
        victim_llm_path,
        cfg.get("temperature", 0.0),
        cfg.get("max_token", 512),
        ollama_model="qwen2.5:7b",
        use_ollama=api_ollama_vllm
    )

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

    def first_method_messages(generate_result):
        if generate_result is None:
            return None
        if isinstance(generate_result, (list, tuple)):
            if not generate_result:
                return None
            return generate_result[0]
        try:
            it = iter(generate_result)
            return next(it, None)
        except TypeError:
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

        try:
            response = victim_llm.response(messages)
        except Exception as e:
            response = f"[ERROR] {e}"

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



if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 run_cypher.py victim_llm_path results_dir dataset_path api_ollama_vllm what_ollama_model")
        sys.exit(1)

    victim_llm_path = sys.argv[1]
    results_dir = sys.argv[2]
    dataset_path = sys.argv[3]
    api_ollama_vllm = str2bool(sys.argv[4].lower())
    what_ollama_model = sys.argv[5]

    run_renellm_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)