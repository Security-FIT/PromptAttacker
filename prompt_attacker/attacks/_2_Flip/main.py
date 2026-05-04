## @file main.py
#  @brief Script for running Flip-based prompt injection attacks
#
#  This script loads a dataset of adversarial prompts and applies
#  a Flip attack strategy against a victim LLM. Results are stored
#  in JSON format for further evaluation.
#
#  @author Bc. Petr Kaška
#  @date 3.1.2026

import os
import json
import pandas
import sys
from attacks.common.llm import LLM
from tqdm import tqdm
from attacks._2_Flip.flip_attack import FlipAttack
from attacks.common.helpers import load_config, str2bool

def run_flip_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):
    """
    @brief Runs the Flip-based prompt injection attack

    @details
    Loads FlipAttack configuration, initializes the victim LLM,
    iterates over a dataset of adversarial prompts, applies the FlipAttack,
    queries the LLM, and stores results in JSON format.

    @param victim_llm_path Path to the victim LLM model
    @param results_dir Directory where results will be stored
    @param dataset_path Path to the CSV dataset with attack goals
    @param api_ollama_vllm Whether to use Ollama / vLLM backend
    @param what_ollama_model Name of the Ollama model
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configFlip.yaml")
    cfg = load_config(config_path)
    cfgFlip = cfg["Flip"]
    print(cfg["Flip"])

    temperature= cfgFlip.get('temperature', 0.0)
    max_token  = cfgFlip.get('max_token', 512)
    flip_mode  = cfgFlip.get('flip_mode', 'FWO')
    cot        = cfgFlip.get('cot', False)
    lang_gpt   = cfgFlip.get('lang_gpt', False)
    few_shot   = cfgFlip.get('few_shot', False)
    begin      = cfgFlip.get('begin', 0)
    end        = cfgFlip.get('end', 13282)


    if dataset_path:
        data_file = dataset_path
        print(f"[INFO] Using data file: {data_file}")

    print(f"[INFO] Loading data from {dataset_path}")
    victim_llm = LLM(model_path=victim_llm_path,
               temperature=temperature,
               max_tokens=max_token,
               ollama_model=what_ollama_model,
               use_ollama=api_ollama_vllm 
               )
    print("[INFO] Initialized LLM client")
    adv_bench = pandas.read_csv(dataset_path)

    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, '_2_flip.json')
    entries = []
    with open(output_file, 'w', encoding='utf-8') as fo:

        for id, harm_prompt in tqdm(enumerate(adv_bench["goal"][begin:end])):
            print(f"[INFO] Processing id {id}: {harm_prompt[:50]}...")
            attack_model = FlipAttack(flip_mode=flip_mode, 
                                    cot=cot, 
                                    lang_gpt=lang_gpt, 
                                    few_shot=few_shot)
            
            log, flip_attack = attack_model.generate(harm_prompt)
            
            llm_response = victim_llm.response(flip_attack)
                        
            entry = {
                'id': id,
                'flip_type': attack_model.flip_mode,
                'original_prompt': log,
                'prompt': flip_attack[-1]['content'],
                'response': llm_response
            }
            entries.append(entry)

        fo.write(json.dumps(entries, ensure_ascii=False) + '\n')
        fo.flush()

    print(f"[INFO] Results saved to {output_file}")

if __name__ == "__main__":
    """
    @brief Script entry point

    @details
    Parses command-line arguments and executes the FlipAttack experiment.
    """
    if len(sys.argv) != 6:
        print("Usage: python3 run_cypher.py victim_llm_path results_dir dataset_path api_ollama_vllm what_ollama_model")
        sys.exit(1)

    victim_llm_path = sys.argv[1]
    results_dir = sys.argv[2]
    dataset_path = sys.argv[3]
    api_ollama_vllm = str2bool(sys.argv[4].lower())
    what_ollama_model = sys.argv[5]

    run_flip_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)