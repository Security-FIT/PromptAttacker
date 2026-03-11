## @file main.py
#  @brief Runner script for Random Search–based prompt injection attack
#
#  This script executes a Random Search attack, a heuristic-based prompt
#  manipulation technique that iteratively modifies the input prompt in order
#  to induce a target LLM to produce a desired response prefix (e.g., "Sure,").
#
#  The attack:
#   - samples random perturbations of the original prompt,
#   - evaluates model responses against a target string,
#   - performs multiple iterations and restarts to increase success probability.
#
#  The script:
#   - loads attack parameters from a YAML configuration file,
#   - iterates over a dataset of adversarial goals,
#   - applies the RandomSearchAttack for each prompt,
#   - queries the victim LLM,
#   - stores results in JSON format for later analysis.
#
#  @author Bc. Petr Kaška
#  @date 30.1.2026
#

import os
import json, sys
import pandas
from tqdm import tqdm

from attacks.common.helpers import load_config, str2bool
from attacks.common.llm import LLM
from attacks._10_Random_Search.randomSearch_attack import RandomSearchAttack


def run_random_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configRandom.yaml")

    cfg = load_config(config_path)
    cfgRS = cfg["RandomSearch"]
    print(cfgRS)

    temperature     = cfgRS.get("temperature", 1.0)
    max_token       = cfgRS.get("max_token", 4096)
    target_str      = cfgRS.get("target_str", "Sure,")
    max_iterations  = cfgRS.get("max_iterations", 3)
    max_restarts    = cfgRS.get("max_restarts", 3)
    max_n_to_change = cfgRS.get("max_n_to_change", 8)
    begin           = cfgRS.get("begin", 0)
    end             = cfgRS.get("end", 519)

    print(f"[INFO] Starting RandomSearch: victim_llm={victim_llm_path}, "
          f"iters={max_iterations}, restarts={max_restarts}, range=[{begin},{end})")

    victim_llm = LLM(model_path=victim_llm_path,
                     temperature=temperature,
                     max_tokens=max_token,
                     ollama_model=what_ollama_model,
                     use_ollama=api_ollama_vllm)
    print("[INFO] Initialized LLM client")

    adv_bench = pandas.read_csv(dataset_path)
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, '_10_randomsearch.json')
    entries = []
    with open(output_file, 'w', encoding='utf-8', errors="replace") as fo:
        for idx, harm_prompt in tqdm(enumerate(adv_bench["goal"][begin:end]), total=end-begin):
            print(f"[INFO] Processing id {idx}: {harm_prompt[:50]}...")

            attacker = RandomSearchAttack(
                victim_llm=victim_llm,
                target_str=target_str,
                max_iterations=max_iterations,
                max_restarts=max_restarts,
                max_n_to_change=max_n_to_change,
                logprob_threshold=-1.0,
                verbose=True,
            )

            out = attacker.generate(harm_prompt)
            prompt_adv = out["adversarial_prompt"]

            llm_response = out["model_response"]  

            entry = {
                "id": idx,
                "original_prompt": harm_prompt,
                "prompt": prompt_adv,
                "response": llm_response,
            }
            entries.append(entry)

        fo.write(json.dumps(entries, ensure_ascii=False) + '\n')
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

    run_random_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)