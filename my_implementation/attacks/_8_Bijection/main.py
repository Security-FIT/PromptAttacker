## @file main.py
#  @brief Runner script for Digit (bijection-based) prompt injection attack
#
#  This script executes the DigitAttack, a bijection-based prompt obfuscation
#  technique where natural language prompts are encoded using a character-to-
#  digit mapping ("Language Alpha") before being submitted to a victim LLM.
#
#  The script:
#   - loads attack parameters from a YAML configuration file,
#   - iterates over a dataset of adversarial goals,
#   - applies the DigitAttack encoding,
#   - queries the victim LLM,
#   - stores results in JSON format for further evaluation.
#
#  @author Bc. Petr Kaška
#  @date 30.1.2026
#

import os
import yaml
import json
import sys

import pandas as pd
from tqdm import tqdm

from attacks._8_Bijection.bijection_digit_attack import DigitAttack
from attacks.common.llm import LLM
from attacks.common.helpers import load_config, str2bool

def run_bijection_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm,what_ollama_model):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configBijection.yaml")
    if not os.path.isfile(config_path):
        sys.exit(f"Konfigurační soubor nenalezen: {config_path}")

    cfg = load_config(config_path)
    cfg_digit = cfg["Digit"]

    temperature = cfg_digit["temperature"]
    max_token = cfg_digit["max_token"]
    begin = cfg_digit.get("begin", 0)
    end = cfg_digit.get("end", None)
    kt = cfg_digit.get("kt", 1)
    kp = cfg_digit.get("kp", 1)

    print(f"[INFO] Spouštím DigitAttack s victim_llm={victim_llm_path}, data_path={dataset_path}")

    if not os.path.isfile(dataset_path):
        sys.exit(f"Soubor s daty nenalezen: {dataset_path}")

    df = pd.read_csv(dataset_path)

    if "goal" not in df.columns:
        sys.exit(f"Chyba: Ve CSV neexistuje sloupec 'goal'.")

    total_rows = len(df)
    if end is None or end > total_rows:
        end = total_rows

    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, "_8_bijection_results.json")

    victim_llm = LLM(
        model_path=victim_llm_path,
        temperature=temperature,
        max_tokens=max_token,
        ollama_model=what_ollama_model,
        use_ollama=api_ollama_vllm, 
    )
    entries = []
    with open(output_file, "w", encoding="utf-8") as fo:
        for idx, row in tqdm(
                enumerate(df["goal"][begin:end]),
                total=end-begin):

            attack_model = DigitAttack(kt=kt, kp=kp)

            messages = attack_model.generate(row)

            try:
                llm_reply = victim_llm.response(messages)
            except Exception as e:
                llm_reply = f"Chyba při volání LLM: {e}"

            encoded_prompt = messages[-1]["content"]

            entry = {
                "id": idx,
                "original_prompt": row,
                "prompt": encoded_prompt,
                "response": llm_reply
            }
            entries.append(entry)

        fo.write(json.dumps(entries, ensure_ascii=False) + "\n")
        fo.flush()

    print(f"[INFO] Výsledky jsou uloženy v {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 run_cypher.py victim_llm_path results_dir dataset_path api_ollama_vllm what_ollama_model")
        sys.exit(1)

    victim_llm_path = sys.argv[1]
    results_dir = sys.argv[2]
    dataset_path = sys.argv[3]
    api_ollama_vllm = str2bool(sys.argv[4].lower())
    what_ollama_model = sys.argv[5]

    run_bijection_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)