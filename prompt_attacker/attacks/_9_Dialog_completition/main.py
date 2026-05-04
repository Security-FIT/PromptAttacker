## @file main.py
#  @brief Runner script for Dialog Completion–based prompt injection attack
#
#  This script executes the DialogCompletionAttack, a dialog-continuation
#  attack that exploits conversational context, persona conditioning, and
#  role-play framing to induce the target LLM to generate a detailed and
#  potentially harmful response.
#
#  The attack is inspired by dialog-completion techniques described in:
#    https://arxiv.org/pdf/2411.06426v1
#
#  The script:
#   - loads attack parameters from a YAML configuration file,
#   - iterates over a dataset of adversarial goals,
#   - embeds each goal into a fabricated prison dialogue,
#   - queries the victim LLM with the combined dialog context,
#   - stores generated responses in JSON format for evaluation.
#
#  @author Bc. Petr Kaška
#  @date 30.1.2026
#
#  Ownership / Contribution statement:
#   - This runner script was fully implemented by Bc. Petr Kaška.
#   - The attack logic is based on dialog-completion ideas from the referenced paper,
#     with original integration into this experimental framework.
#

import os
import json, sys
import pandas as pd
from attacks.common.llm import LLM
from attacks._9_Dialog_completition.dialog_attack import DialogCompletionAttack   
from attacks.common.helpers import load_config, str2bool
from tqdm import tqdm


def run_dialog_attack(victim_llm_path, results_dir, dataset_path,api_ollama_vllm, what_ollama_model):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configDialog.yaml")
    cfg = load_config(config_path)
    cfgSeq = cfg.get("Dialog", {})
    print(cfgSeq)

    temperature  = cfgSeq.get('temperature', 0.0)
    max_tokens   = cfgSeq.get('max_tokens', 512)
    begin        = cfgSeq.get('begin', 0)
    end          = cfgSeq.get('end', None)

    llm_client = LLM(model_path=victim_llm_path, temperature=temperature, max_tokens=max_tokens, 
                     ollama_model=what_ollama_model,
                     use_ollama=api_ollama_vllm)
    print("[INFO] Initialized LLM client")

    print(f"[INFO] Loading data from {dataset_path}")
    df = pd.read_csv(dataset_path)
    if end is None:
        end = len(df)

    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, '_9_dialog_attack.json')
    entries = []

    with open(output_file, 'w', encoding='utf-8') as fo:
        for idx, harm_prompt in tqdm(enumerate(df['goal'][begin:end]), total=end-begin):
            print(f"[INFO] Processing id {idx}: {harm_prompt[:50]}...")
            attack_model = DialogCompletionAttack()
            log, sequence_prompts = attack_model.generate(harm_prompt)

            all_system = " ".join(item['system'] for item in sequence_prompts)

            all_user   = " ".join(item['user']   for item in sequence_prompts)

            print("Combined SYSTEM:\n", all_system)
            print("\nCombined USER:\n", all_user)
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

    run_dialog_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)