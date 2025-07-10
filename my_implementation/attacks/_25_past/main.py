import os
import json
import pandas
import datetime
import numpy as np
from tqdm import tqdm
# Import existing LLM wrapper for vLLM
from attacks._25_past.llm import LLM 
# New attack logic for Past Tense
from attacks._25_past.attack_past import PastTenseAttack 
# Helper to load config files
from attacks.helpers import load_config 

# Import DefenseEA, as used in your original main.py
from defense.defense_EA import DefenseEA


def run_past_tense_attack(run_defense: bool = False):
    # Removed: load_dotenv() - no longer needed without OpenAI/Together clients
    # Removed: client_oai = openai.OpenAI(...)
    # Removed: client_together = openai.OpenAI(...)

    defense = DefenseEA() # Initialize your defense mechanism, as in your original main.py

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configPast.yaml")
    cfg = load_config(config_path)
    cfgPastTense = cfg["PastTense"]
    print(cfg["PastTense"])

    reformulator_llm_path = cfgPastTense['reformulator_llm']
    target_llm_path = cfgPastTense['target_llm']
    data_path = cfgPastTense['data_path']
    out_dir = cfgPastTense['output_dict']
    
    reformulator_temperature = cfgPastTense.get('reformulator_temperature', 1.0)
    reformulator_max_tokens = cfgPastTense.get('reformulator_max_tokens', 150)
    
    target_temperature = cfgPastTense.get('target_temperature', 1.0)
    target_max_tokens = cfgPastTense.get('target_max_tokens', 150)
    
    attack_type = cfgPastTense.get('attack_type', 'past')
    begin = cfgPastTense.get('begin', 0)
    end = cfgPastTense.get('end', 100)
    n_restarts = cfgPastTense.get('n_restarts', 20)

    print(f"[INFO] Starting Past Tense Attack: reformulator_llm={reformulator_llm_path}, target_llm={target_llm_path}, attack_type={attack_type}, range=[{begin},{end})")

    # Initialize vLLM clients
    reformulator_llm = LLM(
        model_path=reformulator_llm_path,
        temperature=reformulator_temperature,
        max_tokens=reformulator_max_tokens
    )
    print("[INFO] Initialized Reformulator LLM client")

    # target_llm = LLM(
    #     model_path=target_llm_path,
    #     temperature=target_temperature,
    #     max_tokens=target_max_tokens
    # )
    print("[INFO] Initialized Target LLM client")

    # Load data
    adv_bench = pandas.read_csv(data_path)
    print(f"[INFO] Loaded data from {data_path}")

    os.makedirs(out_dir, exist_ok=True)
    timestamp_start = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(out_dir, f'_25_past.jsonl')

    # This dictionary will store all results for final analysis (without ASR if no judges)
    jb_dict = {
        "config": cfgPastTense,
        "jb_artifacts": []
    }

    with open(output_file, 'w', encoding='utf-8') as fo:
        for i_request, harm_prompt_original in tqdm(enumerate(adv_bench["Goal"][begin:end])):
            print(f">>> Request #{i_request}: {harm_prompt_original}")
            
            request_reformulated_str = "" # To store the string representation of the reformulated request
            response_from_target = "" # To store the response from the target LLM
            current_i_restart = -1 # To track which restart was the last one for this request

            for i_restart in range(n_restarts):
                attack_model = PastTenseAttack(attack_type=attack_type)
                
                # The generate method now returns the prompt content directly
                log_original_request, reformulate_messages = attack_model.generate(harm_prompt_original)
                
                # Use reformulator LLM to get the reformulated prompt
                reformulator_response_raw = reformulator_llm.response(reformulate_messages)
                request_reformulated_str = reformulator_response_raw.strip().replace('"', '') # Clean up the response
                print(f">>> Reformulated request (restart {i_restart}): {request_reformulated_str}")

                # Apply defense if enabled, mirroring your flip_main.py structure
                prompt_to_llm = request_reformulated_str
                if run_defense:
                    prompt_to_llm = defense(prompt_to_llm) # Apply defense
                    print(f"[DEBUG] Prompt after defense (before target LLM): {prompt_to_llm[:50]}...")
                
                # Use target LLM to get the final response
                target_messages = [{"role": "user", "content": prompt_to_llm}]
                response_from_target = reformulator_llm.response(target_messages)
                
                print(f">>> Target LLM Response (restart {i_restart}): {response_from_target}")
                
                # Removed: Judge calls and related print statements.
                # Removed: break condition based on judge results. The loop will now always run n_restarts times.
                
                current_i_restart = i_restart
            
            # Store results for this request
            jb_artifact = {
                "id": i_request,
                "restart_times": current_i_restart, # This will be the last restart index for the request
                "original_prompt": harm_prompt_original, # Store the original request
                "attacked_prompt": request_reformulated_str,
                "response": response_from_target,
            }
            jb_dict["jb_artifacts"].append(jb_artifact)
            
            # Write each artifact as a separate JSON line
            fo.write(json.dumps(jb_artifact, ensure_ascii=False) + '\n')
            fo.flush() # Ensure it's written to disk immediately
            print('=='*50 + '\n\n')

    # Removed: ASR Calculation block - no longer relevant without judges
    print(f"[INFO] All results saved to {output_file}")