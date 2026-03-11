## @file main.py
#  @brief Runner script for rewrite-based prompt transformation attack
#
#  This script evaluates a rewrite-based jailbreak attack where a harmful or
#  restricted user goal is first rewritten using a transformation template
#  and then submitted to a victim large language model.
#
#  The rewrite attack aims to preserve the semantic intent of the original
#  prompt while altering its surface form, potentially bypassing safety
#  filters and alignment mechanisms that rely on lexical or structural cues.
#
#  The script:
#   - Loads rewrite parameters from `configRewrite.yaml`
#   - Reads a CSV dataset containing adversarial goals (column: `goal`)
#   - Applies a rewrite transformation via `RewriteAttack`
#   - Queries the victim LLM with the rewritten prompt
#   - Stores original prompts, rewritten prompts, and model responses
#     in JSON format
#
#  @author Bc. Petr Kaška
#  @date 1.2.2026
#
#  Ownership / Contribution statement:
#   - This runner script was fully implemented by Bc. Petr Kaška.
#   - The experiment orchestration, rewrite attack invocation, dataset slicing,
#     LLM querying logic, and result serialization are original work by the author.
#   - The rewrite attack module (`RewriteAttack`) is integrated as a component
#     within the author's experimental framework.
#

import os
import json, sys
import pandas
from tqdm import tqdm

from attacks.common.llm import LLM
from attacks._15_Rewrite.attack_rewrite import RewriteAttack
from attacks.common.helpers import load_config, str2bool

def run_rewrite_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configRewrite.yaml")
    cfg         = load_config(config_path)["Rewrite"]

    temperature     = cfg.get("temperature", 0.0)
    max_token       = cfg.get("max_token", 512)
    rewrite_tmpl    = cfg["rewrite_template"]
    begin           = cfg.get("begin", 0)
    end             = cfg.get("end", 100)

    print(f"[INFO] Rewrite: model={victim_llm_path}, rows=[{begin},{end})")

    adv_bench = pandas.read_csv(dataset_path)
    llm = LLM(model_path=victim_llm_path,
              temperature=temperature,
              max_tokens=max_token,
              ollama_model=what_ollama_model,
              use_ollama=api_ollama_vllm)

    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, '_15_rewrite.json')
    entries = []
    with open(output_file, "w", encoding="utf-8") as fo:
        for idx, harm_prompt in tqdm(
            enumerate(adv_bench["goal"][begin:end]), total=end - begin
        ):
            attack_model = RewriteAttack(rewrite_template=rewrite_tmpl)

            log, attack_msgs = attack_model.generate(harm_prompt)

            llm_response = llm.response(attack_msgs)

            entry = {
                "id": idx,
                "original_prompt": log,
                "prompt": attack_msgs[-1]["content"],
                "response": llm_response,
            }
            entries.append(entry)

        fo.write(json.dumps(entries, ensure_ascii=False) + "\n")
        fo.flush()

    print(f"[INFO] Results saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 run_cypher.py victim_llm_path results_dir dataset_path api_ollama_vllm what_ollama_model")
        sys.exit(1)

    victim_llm_path = sys.argv[1]
    results_dir = sys.argv[2]
    dataset_path = sys.argv[3]
    api_ollama_vllm = str2bool(sys.argv[4].lower())
    what_ollama_model = sys.argv[5]

    run_rewrite_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)