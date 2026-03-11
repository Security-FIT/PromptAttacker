## @file main.py
#  @brief Script for running Cypher-based prompt injection attacks
#
#  This script loads a dataset of adversarial prompts and applies
#  a Cypher attack strategy against a victim LLM. Results are stored
#  in JSON format for further evaluation.
#
#  @author Bc.Petr Kaška
#  @date 3.1.2026

import sys
import os
import json
import pandas
from attacks.common.llm import LLM
from tqdm import tqdm
from attacks._1_Cypher.cypher_attack import CypherAttack
from attacks.common.helpers import load_config, str2bool


def run_cypher_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):
    """
    @brief Executes a Cypher attack against a victim LLM

    Loads configuration parameters, initializes the victim LLM,
    iterates over a dataset of adversarial prompts, and applies
    the Cypher attack strategy.

    @param victim_llm_path Path to the victim LLM model
    @param results_dir Directory where results will be saved
    @param dataset_path Path to the CSV dataset containing attack goals
    @param api_ollama_vllm Enables Ollama / vLLM backend
    @param what_ollama_model Name of the Ollama model to use

    @return None
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configCypher.yaml")
    cfg = load_config(config_path)
    cfgCypher = cfg["Cypher"]
    temperature= cfgCypher.get('temperature', 0.0)
    max_token  = cfgCypher.get('max_token', 512)
    cypher_mode  = cfgCypher.get('cypher_mode', 'WSWR')
    few_shot   = cfgCypher.get('few_shot', False)
    begin      = cfgCypher.get('begin', 0)
    end        = cfgCypher.get('end', 13282)

    if dataset_path:
        print(f"[INFO] Using data file: {dataset_path}")

    print(f"[INFO] Loading data from {dataset_path}")
    victim_llm = LLM(model_path=victim_llm_path,
               temperature=temperature,
               max_tokens=max_token,
               ollama_model=what_ollama_model,
               use_ollama=api_ollama_vllm, 
               )
    print("[INFO] Initialized LLM client")
    adv_bench = pandas.read_csv(dataset_path)

    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, '_1_cypher.json')
    entries = []
    with open(output_file, 'w', encoding='utf-8') as fo:

        for id, harm_prompt in tqdm(enumerate(adv_bench["goal"][begin:end])):
            print(f"[INFO] Processing id {id}")
            attack_model = CypherAttack(cypher_mode=cypher_mode, 
                                    few_shot=few_shot,
                                    victim_llm=victim_llm)
            
            log, cypher_attack = attack_model.generate(harm_prompt)
            
            llm_response = victim_llm.response(cypher_attack)
            
            entry = {
                'id': id,
                'cypher_type': attack_model.cypher_mode,
                'original_prompt': log,
                'prompt': cypher_attack[-1]['content'],
                'response': llm_response
            }
            entries.append(entry)

        fo.write(json.dumps(entries, ensure_ascii=False) + '\n')
        fo.flush()

    print(f"[INFO] Results saved to {output_file}")


if __name__ == "__main__":
    """
    @brief Entry point of the script

    Parses command-line arguments and runs the Cypher attack.
    """
    if len(sys.argv) != 6:
        print("Usage: python3 run_cypher.py victim_llm_path results_dir dataset_path api_ollama_vllm what_ollama_model")
        sys.exit(1)

    victim_llm_path = sys.argv[1]
    results_dir = sys.argv[2]
    dataset_path = sys.argv[3]
    api_ollama_vllm = str2bool(sys.argv[4].lower())
    what_ollama_model = sys.argv[5]

    run_cypher_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)