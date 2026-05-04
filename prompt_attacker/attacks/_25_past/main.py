## @file main.py
#  @brief Runner script for the Past Tense (temporal reformulation) jailbreak attack
#
#  This script evaluates a Past Tense prompt attack, where the original harmful
#  user request is reformulated into a past-tense or temporally shifted version
#  using a dedicated reformulator LLM. The reformulated prompt is then submitted
#  to the target (victim) LLM to test whether temporal reframing weakens safety
#  alignment or policy enforcement.
#
#  @author Bc. Petr Kaška
#  @date 1.2.2026
#
#  Ownership / Contribution statement:
#   - This runner script was fully designed and implemented by Bc. Petr Kaška.
#   - The orchestration logic, dual-LLM setup, restart sampling loop,
#     dataset handling, logging, and result serialization are original work
#     by the author.
#   - The prompt construction logic for temporal reformulation is delegated to
#     `PastTenseAttack` and integrated into the author’s experimental framework.
#

import os
import json
import pandas, sys
import datetime
import numpy as np
from tqdm import tqdm
from attacks.common.llm import LLM 
from attacks._25_past.attack_past import PastTenseAttack 
from attacks.common.helpers import load_config, str2bool

def run_past_tense_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configPast.yaml")
    cfg = load_config(config_path)
    cfgPastTense = cfg["PastTense"]
    print(cfg["PastTense"])

    reformulator_llm_path = cfgPastTense['reformulator_llm']
    
    reformulator_temperature = cfgPastTense.get('reformulator_temperature', 1.0)
    reformulator_max_tokens = cfgPastTense.get('reformulator_max_tokens', 150)
    
    target_temperature = cfgPastTense.get('target_temperature', 1.0)
    target_max_tokens = cfgPastTense.get('target_max_tokens', 150)
    
    attack_type = cfgPastTense.get('attack_type', 'past')
    begin = cfgPastTense.get('begin', 0)
    end = cfgPastTense.get('end', 100)
    n_restarts = cfgPastTense.get('n_restarts', 20)

    print(f"[INFO] Starting Past Tense Attack: reformulator_llm={reformulator_llm_path}, target_llm={victim_llm_path}, attack_type={attack_type}, range=[{begin},{end})")

    reformulator_llm = LLM(
        model_path=reformulator_llm_path,
        temperature=reformulator_temperature,
        max_tokens=reformulator_max_tokens,
        ollama_model="qwen2.5:7b",
        use_ollama=True,
    )
    print("[INFO] Initialized Reformulator LLM client")

    target_llm = LLM(
        model_path=victim_llm_path,
        temperature=target_temperature,
        max_tokens=target_max_tokens,
        ollama_model=what_ollama_model,
        use_ollama=api_ollama_vllm, 
    )
    print("[INFO] Initialized Target LLM client")

    adv_bench = pandas.read_csv(dataset_path)
    print(f"[INFO] Loaded data from {dataset_path}")

    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f'_25_past.json')

    entries = []

    with open(output_file, 'w', encoding='utf-8') as fo:
        for i_request, harm_prompt_original in tqdm(enumerate(adv_bench["goal"])):
            print(f">>> Request #{i_request}: {harm_prompt_original}")
            
            request_reformulated_str = ""
            response_from_target = "" 
            current_i_restart = -1 

            for i_restart in range(n_restarts):
                attack_model = PastTenseAttack(attack_type=attack_type)
                
                log_original_request, reformulate_messages = attack_model.generate(harm_prompt_original)
                
                reformulator_response_raw = reformulator_llm.response(reformulate_messages)
                request_reformulated_str = reformulator_response_raw.strip().replace('"', '') 
                print(f">>> Reformulated request (restart {i_restart}): {request_reformulated_str}")

                prompt_to_llm = request_reformulated_str
                target_messages = [{"role": "user", "content": prompt_to_llm}]
                response_from_target = target_llm.response(target_messages)
                
                print(f">>> Target LLM Response (restart {i_restart}): {response_from_target}")
                
                current_i_restart = i_restart
            
            jb_artifact = {
                "id": i_request,
                "restart_times": current_i_restart, 
                "original_prompt": harm_prompt_original,
                "prompt": request_reformulated_str,
                "response": response_from_target,
            }
            entries.append(jb_artifact)
            
        fo.write(json.dumps(entries, ensure_ascii=False) + '\n')
        fo.flush() 
        print('=='*50 + '\n\n')

    print(f"[INFO] All results saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 run_cypher.py victim_llm_path results_dir dataset_path api_ollama_vllm what_ollama_model")
        sys.exit(1)

    victim_llm_path = sys.argv[1]
    results_dir = sys.argv[2]
    dataset_path = sys.argv[3]
    api_ollama_vllm = str2bool(sys.argv[4].lower())
    what_ollama_model = sys.argv[5]

    run_past_tense_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)