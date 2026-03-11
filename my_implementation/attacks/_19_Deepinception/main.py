## @file main.py
#  @brief Runner script for the DeepInception prompt-based jailbreak attack
#
#  This script evaluates the DeepInception attack, a role-playing and
#  narrative-layer–based prompt injection technique. The attack embeds a
#  harmful user request into a multi-layer fictional scenario with multiple
#  characters, gradually distancing the request from its original form in
#  order to weaken safety alignment.
#
#  The script:
#   - Loads experiment parameters from `configInception.yaml`
#   - Reads a CSV dataset containing adversarial goals (column: `goal`)
#   - Generates DeepInception-style prompts using `DeepInceptionAttack`
#   - Queries the victim LLM with the generated messages
#   - Stores original prompts, transformed prompts, and model responses
#     in JSON format
#
#  @author Bc. Petr Kaška
#  @date 1.2.2026
#
#  Ownership / Contribution statement:
#   - This runner script was fully implemented by Bc. Petr Kaška.
#   - The experiment orchestration, configuration handling, dataset slicing,
#     attack invocation, LLM querying, and result serialization are original
#     work by the author.
#   - The DeepInception attack logic is integrated as an external attack
#     module within the author’s experimental framework.
#

import os
import json, sys
import pandas
from tqdm import tqdm

from attacks.common.llm import LLM          
from attacks._19_Deepinception.deep_inception_attack import DeepInceptionAttack
from attacks.common.helpers import load_config, str2bool

def run_inception_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configInception.yaml")
    cfg         = load_config(config_path)["DeepInception"]

    temperature  = cfg.get("temperature", 0.0)
    max_token    = cfg.get("max_token", 512)
    scene        = cfg.get("scene", "science fiction")
    char_num     = cfg.get("character_number", 5)
    layer_num    = cfg.get("layer_number", 5)
    begin        = cfg.get("begin", 0)
    end          = cfg.get("end", 100)

    print(f"[INFO] DeepInception: model={victim_llm_path}, rows=[{begin},{end})")

    adv_bench = pandas.read_csv(dataset_path)
    llm = LLM(model_path=victim_llm_path,
              temperature=temperature,
              max_tokens=max_token,
              ollama_model=what_ollama_model,
              use_ollama=api_ollama_vllm)

    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, '_19_deep_inception.json')
    entries = []
    with open(output_file, "w", encoding="utf-8") as fo:
        for idx, harm_prompt in tqdm(
            enumerate(adv_bench["goal"][begin:end]), total=end - begin
        ):
            attack_model = DeepInceptionAttack(
                scene=scene,
                character_number=char_num,
                layer_number=layer_num,
            )

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
        print("Usage: python3 run_inception_attack.py victim_llm_path results_dir dataset_path api_ollama_vllm what_ollama_model")
        sys.exit(1)

    victim_llm_path = sys.argv[1]
    results_dir = sys.argv[2]
    dataset_path = sys.argv[3]
    api_ollama_vllm = str2bool(sys.argv[4].lower())
    what_ollama_model = sys.argv[5]

    run_inception_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)