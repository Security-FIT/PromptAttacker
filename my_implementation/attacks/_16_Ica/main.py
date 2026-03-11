## @file main.py
#  @brief Runner script for ICA (Implicit Chain-of-Thought) jailbreak attack
#
#  This script evaluates the ICA (Implicit Chain-of-Thought) attack, which
#  constructs prompts designed to elicit hidden or implicit reasoning steps
#  from the victim LLM. The attack optionally enables a chain-of-thought (CoT)
#  style prompting strategy that may weaken safety alignment by encouraging
#  internal reasoning before policy enforcement.
#
#  The script:
#   - Loads experiment parameters from `configIca.yaml`
#   - Reads a CSV dataset containing adversarial goals (column: `goal`)
#   - Generates ICA-style prompts using `ICAAttack`
#   - Queries the victim LLM with the generated messages
#   - Stores original prompts, ICA prompts, and model responses in JSON format
#
#  This runner is intended for systematic evaluation of reasoning-based
#  jailbreak attacks against aligned large language models.
#
#  @author Bc. Petr Kaška
#  @date 1.2.2026
#
#  Ownership / Contribution statement:
#   - This runner script was fully implemented by Bc. Petr Kaška.
#   - The experiment orchestration, ICA invocation, dataset slicing,
#     error handling, LLM querying, and result serialization are
#     original work by the author.
#   - The ICA attack module is integrated into the author's
#     experimental evaluation framework.


import os, sys,json, pandas as pd
from pathlib import Path
from tqdm import tqdm

from attacks._16_Ica.ica_attack import ICAAttack
from attacks.common.llm import LLM
from attacks.common.helpers import load_config, str2bool

def run_ica_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(script_dir + "/configIca.yaml")["ICA"]

    temperature = cfg.get("temperature", 0.0)
    max_token   = cfg.get("max_token", 512)
    cot         = cfg.get("cot", False)
    begin       = cfg.get("begin", 0)
    end         = cfg.get("end", None)

    print(f"[INFO] ICA-Attack  •  model={victim_llm_path}  •  COT={cot}")

    victim   = LLM(victim_llm_path, temperature, max_token, what_ollama_model, api_ollama_vllm)
    attacker = ICAAttack(cot=cot)

    df = pd.read_csv(dataset_path)

    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, '_16_ica.json')

    entries = []
    with open(out_file, "w", encoding="utf-8") as fo:
        for idx, row in tqdm(
            enumerate(df["goal"][begin:end]), total=end - begin
        ):
            log, msgs = attacker.generate(row)

            try:
                reply = victim.response(msgs)
            except Exception as e:
                reply = f"[ERROR] {e}"

            entry = {
                "id": idx + begin,
                "original_prompt": row,
                "prompt": msgs[-1]["content"],
                "response": reply
            }
            entries.append(entry)
        fo.write(json.dumps(entries, ensure_ascii=False) + "\n")
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

    run_ica_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)