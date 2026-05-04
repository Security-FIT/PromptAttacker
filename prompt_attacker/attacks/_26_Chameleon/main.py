## @file main.py
#  @brief Runner script for the CodeChameleon (“Chameleon”) jailbreak attack using a fixed `binary_tree` cipher.
#
#  This script serves as a minimal launcher for the Chameleon-style attack variant
#  integrated in the author’s experimental framework. The interface is intentionally
#  kept equivalent to other runners (e.g., `run_flip_attack()`), so it can be used
#  interchangeably in batch experiments.
#
#  Notes:
#   - The cipher is currently fixed to `binary_tree` to match the experimental setup
#     and reproduce prior results deterministically.
#   - Prompt construction and encryption are delegated to the attack modules:
#     `attacks._26_Chameleon.encrypt` and `attacks._26_Chameleon.template`.
#
#  @author Bc. Petr Kaška
#  @date 1.2.2026
#
#  Ownership / Contribution statement:
#   - This runner script was fully designed and implemented by Bc. Petr Kaška.
#   - The orchestration logic (config loading, dataset slicing, LLM invocation loop,
#     progress tracking, filesystem output management, and JSON serialization) is
#     original work by the author.
#   - The encryption method and prompt templates are delegated to:
#       - `get_encrypted_query()` (attack-specific encryption)
#       - `get_prompts_text()` / `get_prompts_code()` (attack-specific templates)
#     and are integrated here as part of the author’s experimental framework.

import os, sys
import json
import pandas as pd
from tqdm import tqdm

from attacks.common.llm import LLM
from attacks.common.helpers import load_config, str2bool
from attacks._26_Chameleon.encrypt import get_encrypted_query
from attacks._26_Chameleon.template import get_prompts_code, get_prompts_text

def run_chameleon_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configChameleon.yaml")
    cfg         = load_config(config_path)
    cfgCham     = cfg["Chameleon"]          

    print("[CONFIG]", cfgCham)

    temperature   = cfgCham.get("temperature", 1.0)
    max_token     = cfgCham.get("max_token",  512)
    begin         = cfgCham.get("begin", 0)
    end           = cfgCham.get("end", None)          

    victim_llm = LLM(
        model_path=victim_llm_path,
        temperature=temperature,
        max_tokens=max_token,
        ollama_model=what_ollama_model,
        use_ollama=api_ollama_vllm,
    )
    print("[INFO] LLM inicializován")

    df = pd.read_csv(dataset_path)
    if "problem" in df.columns:
        prompts_series = df["problem"]
    elif "goal" in df.columns:
        prompts_series = df["goal"]
    else:
        raise ValueError("CSV musí obsahovat sloupec 'problem' nebo 'goal'.")

    if end is None or end > len(prompts_series):
        end = len(prompts_series)

    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, "_26_Chameleon.json")

    entries = []
    with open(output_file, "w", encoding="utf-8") as fo:
        for idx, original_problem in tqdm(
            enumerate(prompts_series[begin:end]), total=end - begin
        ):
            print(f"[INFO] → #{idx}: {original_problem[:60]}…")

            encrypted_problem = get_encrypted_query(original_problem, "binary_tree")

            prompt = get_prompts_text([encrypted_problem], "binary_tree")[0]

            llm_response = victim_llm.response([{"role": "user", "content": prompt}])

            entry = {
                "id": idx,
                "original_prompt": original_problem,
                "prompt": prompt,
                "response": llm_response,
            }
            entries.append(entry)
        fo.write(json.dumps(entries, ensure_ascii=False) + "\n")
        fo.flush()

    print(f"[INFO] Hotovo – uložené výsledky: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 run_cypher.py victim_llm_path results_dir dataset_path api_ollama_vllm what_ollama_model")
        sys.exit(1)

    victim_llm_path = sys.argv[1]
    results_dir = sys.argv[2]
    dataset_path = sys.argv[3]
    api_ollama_vllm = str2bool(sys.argv[4].lower())
    what_ollama_model = sys.argv[5]

    run_chameleon_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)