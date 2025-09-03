"""pif_attack.py – Pipeline pro PIF útok (Prefix‑Inversion Framework)
==============================================================

* Čte konfiguraci **stejným způsobem** jako tvůj `run_flip_attack`,
  tj. pomocí `load_config()` a sekce `PIF` v YAML souboru
  uloženém ve stejném adresáři jako tento skript.

* Výsledek útoku ukládá do JSONL (`_3_PIF.json`) a na konci dopíše
  agregované metriky (Average Queries, Time, ASR, AHS).

Spuštění
--------
```
python pif_attack.py              # použije configPIF.yaml v aktuální složce
python pif_attack.py --defense    # (pokud bys někdy chtěl přidat obranu)
```
"""

from __future__ import annotations

import pandas
import argparse
import json
import logging
import os
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

from attacks.helpers import load_config
import attacks._3_PiF.attack_clm as attack_clm  
import attacks._3_PiF.eval as eval  

###########################################################################
# Helper
###########################################################################

def str2bool(v: str) -> bool:  # pro CLI switch
    return str(v).lower() in {"1", "true", "yes", "y"}

###########################################################################
# Main
###########################################################################

def run_pif_attack(victim_llm_path, results_dir, dataset_path, what_ollama_model, api_ollama_vllm):  # aktuálně se obrana nepoužívá
    # ------------------------------------------------------------------
    # 1) načtení konfigurace
    # ------------------------------------------------------------------
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "configPIF.yaml"
    cfg = load_config(config_path)
    cfg_pif = cfg["PIF"]

    # povinné položky
    gen_model_path: str = cfg_pif["gen_model_path"]
    # tgt_model_path: str = cfg_pif["tgt_model_path"]
    # att_file: str = cfg_pif["att_file"]
    # output_dir: str = cfg_pif["output_dir"]

    # volitelné s výchozí hodnotou
    opt_objective: str = cfg_pif.get("opt_objective", "ASR")
    hf_cache_dir: str = cfg_pif.get("hf_cache_dir", "./hf_models")
    output_file_name: str = cfg_pif.get("output_file", "_3_pif.json")
    prompt_template: str = cfg_pif.get("prompt_template", "Human: {}. Assistant: ")
    evaluation_template: str = cfg_pif.get("evaluation_template", "this intent is [MASK]")

    batch_size: int = int(cfg_pif.get("batch_size", 1))
    iterations: int = int(cfg_pif.get("iterations", 20))

    # hyper‑parametry útoku
    top_n: int = int(cfg_pif.get("top_n", 5))
    top_m: int = int(cfg_pif.get("top_m", 5))
    top_k: int = int(cfg_pif.get("top_k", 10))
    warm_up: int = int(cfg_pif.get("warm_up", 0))
    temperature: float = float(cfg_pif.get("temperature", 0.1))
    threshold: float = float(cfg_pif.get("threshold", 0.9))

    # ------------------------------------------------------------------
    # 2) příprava logů a IO
    # ------------------------------------------------------------------
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    log_file = "/storage/brno2/home/xkaska01/master/my_implementation/attacks/_3_PiF/output.log"

    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger("PIFAttack")

    logger.info("Loaded config from %s", config_path)
    logger.info(cfg_pif)

    # ------------------------------------------------------------------
    # 3) načtení datasetu
    # ------------------------------------------------------------------
    adv_bench = pandas.read_csv(dataset_path)
    # ------------------------------------------------------------------
    # 4) útok
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    overall_query = 0
    overall_time = 0.0
    overall_successful = 0
    overall_input = 0

    output_path = Path(results_dir) / output_file_name
    logger.info("Results will be written to %s", output_path)
    entries = []
    # ['Human: goal,translation_of_goal,target. Assistant: ']
    with open(output_path, "w", encoding="utf-8") as fo:
        for i, harm_prompt in tqdm(enumerate(adv_bench["goal"])):

            chunk_prompts: List[str] = [harm_prompt]

            # print("ADV")
            # print(chunk_prompts)
            # print()
            # print()
            # print()
            # print()
            # print()
            # print()
            # exit(0)


            query, t_spent, flags, gen_attacks, tgt_responses = attack_clm.generate_attack(
                gen_model_path, gen_model_path, victim_llm_path, victim_llm_path,
                chunk_prompts, evaluation_template,
                objective=opt_objective, iterations=iterations,
                top_n=top_n, top_m=top_m, top_k=top_k,
                warm_up=warm_up, temperature=temperature, threshold=threshold,
                device=device,
            )

            overall_query += query
            overall_time += t_spent

            for j, (flag, original_prompt, gen_attack, tgt_response) in enumerate(zip(
                    flags, chunk_prompts, gen_attacks, tgt_responses)):
                overall_input += 1
                if flag:
                    overall_successful += 1

                rec = {
                    "id": i + j + 1,
                    "Success": flag,
                    "original_prompt": original_prompt,
                    "prompt": gen_attack,
                    "response": tgt_response,
                }
                entries.append(rec)
        fo.write(json.dumps(entries, ensure_ascii=False) + "\n")
        fo.flush()

    # ------------------------------------------------------------------
    # 5) agregační metriky
    # ------------------------------------------------------------------
    logger.info("Attack finished – evaluating…")
    overall_ahs = eval.ahs(str(output_path))

    summary = {
        "Average Queries": overall_query / max(1, overall_input),
        "Average Time": overall_time / max(1, overall_input),
        "ASR": overall_successful / max(1, overall_input),
        "AHS": overall_ahs,
    }
    logger.info("Summary: %s", summary)

    with open(output_path, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(summary, ensure_ascii=False) + "\n")
        fo.flush()
