## @file main.py
#  @brief Script for running Sequential-based prompt injection attacks
#
#  This script loads a dataset of adversarial prompts and applies a Sequential
#  attack strategy against a victim LLM. The attack decomposes the prompt into
#  multiple sequential subtasks, merges them into a single (system, user) prompt,
#  and queries the target model. Results are stored in JSON format for further
#  evaluation.
#
#  @author Bc. Petr Kaška
#  @date 3.1.2026

import os
import json, sys
import pandas as pd
from attacks.common.llm import LLM
from attacks._6_Sequential.sequential_attack import SequentialAttack   
from attacks.common.helpers import load_config, str2bool
from tqdm import tqdm


def run_sequential_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):
    """
    @brief Executes the Sequential attack on a dataset of adversarial prompts

    @details
    Loads configuration parameters, initializes the victim LLM, iterates through the
    dataset, generates a sequence of subtasks using SequentialAttack, merges all
    subtask messages into a single system/user prompt, queries the model, and stores
    results to a JSON file.

    @param victim_llm_path Path to the victim LLM model
    @param results_dir Directory where results will be saved
    @param dataset_path Path to the CSV dataset containing attack goals (column "goal")
    @param api_ollama_vllm Enables Ollama / vLLM backend
    @param what_ollama_model Name of the Ollama model to use

    @return None
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configSequential.yaml")
    cfg = load_config(config_path)
    cfgSeq = cfg.get("Sequential", {})

    temperature  = cfgSeq.get('temperature', 0.0)
    max_tokens   = cfgSeq.get('max_tokens', 512)
    steps        = cfgSeq.get('steps', 3)
    begin        = cfgSeq.get('begin', 0)
    end          = cfgSeq.get('end', None)

    print("[INFO] Initialized LLM client")
    llm_client = LLM(model_path=victim_llm_path,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    ollama_model=what_ollama_model,
                    use_ollama=api_ollama_vllm)

    print(f"[INFO] Loading data from {dataset_path}")
    df = pd.read_csv(dataset_path)
    if end is None:
        end = len(df)

    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, '_6_sequential_attack.json')
    entries = []
    with open(output_file, 'w', encoding='utf-8') as fo:
        for idx, harm_prompt in tqdm(enumerate(df['goal'][begin:end]), total=end-begin):
            print(f"[INFO] Processing id {idx}: {harm_prompt[:50]}...")
            
            attack_model = SequentialAttack(steps)
            log, sequence_prompts = attack_model.generate(harm_prompt)

            all_system = " ".join(item['system'] for item in sequence_prompts)
            all_user   = " ".join(item['user']   for item in sequence_prompts)

            response = llm_client.response([
                {'role': 'system', 'content': all_system},
                {'role': 'user', 'content': all_user}
            ])

            entry = {
                'id': idx,
                'original_prompt': log,
                'prompt': all_user,
                'response': response
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

    run_sequential_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)