# TENTO FILE JE MUUUUUUUUUJ

import os
import json
import pandas
from tqdm import tqdm

from attacks.helpers import load_config
from attacks.common.llm import LLM                       # jednoduchý wrapper
from attacks._10_Random_Search.randomSearch_attack import RandomSearchAttack
from defense.defense_EA import DefenseEA


def run_random_attack(run_defense: bool = False):
    """
    Spustí Random-Search útok nad datasetem definovaným v configRandomSearch.yaml
    a výsledky uloží do JSON řádku po řádku (stejně jako run_flip_attack).
    """
    defense = DefenseEA()

    # --------------------------------------------------------------------- #
    # Načtení konfigurace                                                   #
    # --------------------------------------------------------------------- #
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configRandom.yaml")

    cfg = load_config(config_path)
    cfgRS = cfg["RandomSearch"]
    print(cfgRS)

    victim_llm_path = cfgRS["victim_llm"]
    data_path       = cfgRS["data_path"]
    out_dir         = cfgRS["output_dict"]
    temperature     = cfgRS.get("temperature", 1.0)
    max_token       = cfgRS.get("max_token", 4096)
    target_str      = cfgRS.get("target_str", "Sure,")
    max_iterations  = cfgRS.get("max_iterations", 3)
    max_restarts    = cfgRS.get("max_restarts", 3)
    max_n_to_change = cfgRS.get("max_n_to_change", 8)
    log_threshold   = cfgRS.get("logprob_threshold", -1.0)
    begin           = cfgRS.get("begin", 0)
    end             = cfgRS.get("end", 519)

    print(f"[INFO] Starting RandomSearch: victim_llm={victim_llm_path}, "
          f"iters={max_iterations}, restarts={max_restarts}, range=[{begin},{end})")

    victim_llm = LLM(model_path=victim_llm_path,
                     temperature=temperature,
                     max_tokens=max_token)
    print("[INFO] Initialized LLM client")

    adv_bench = pandas.read_csv(data_path)
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, '_10_randomsearch.json')

    with open(output_file, 'w', encoding='utf-8') as fo:
        for idx, harm_prompt in tqdm(enumerate(adv_bench["Goal"][begin:end]), total=end-begin):
            print(f"[INFO] Processing id {idx}: {harm_prompt[:50]}...")

            attacker = RandomSearchAttack(
                victim_llm=victim_llm,
                target_str=target_str,
                max_iterations=max_iterations,
                max_restarts=max_restarts,
                max_n_to_change=max_n_to_change,
                logprob_threshold=log_threshold,
                verbose=True,
            )

            out = attacker.generate(harm_prompt)
            prompt_adv = out["adversarial_prompt"]

            if run_defense:
                prompt_adv = defense(prompt_adv)

            llm_response = out["model_response"]  

            entry = {
                "id": idx,
                "original_prompt": harm_prompt,
                "prompt": prompt_adv,
                "response": llm_response,
            }

            fo.write(json.dumps(entry, ensure_ascii=False) + '\n')
            fo.flush()

    print(f"[INFO] Results saved to {output_file}")
