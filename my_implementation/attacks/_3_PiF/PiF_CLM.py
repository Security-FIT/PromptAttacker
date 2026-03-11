## @file run_pif.py
#  @brief Script for running PiF-based prompt injection attacks
#
#  This script loads a dataset of adversarial prompts and applies a PiF attack
#  strategy against a victim LLM using a configurable generation model.
#  Results are stored in JSON format and summarized with evaluation metrics.
#
#  @author Bc. Petr Kaška
#  @date 3.1.2026

from __future__ import annotations

import pandas, sys
import argparse
import json
import logging
import os
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

from attacks.common.helpers import load_config, str2bool
import attacks._3_PiF.attack_clm as attack_clm  
import attacks._3_PiF.eval as eval  


def run_pif_attack(victim_llm_path, results_dir, dataset_path, what_ollama_model, api_ollama_vllm):  
    """
    @brief Executes the PiF attack over a dataset of adversarial prompts

    @details
    This function:
     - loads PiF configuration from configPIF.yaml,
     - initializes logging and selects compute device (CUDA if available),
     - iterates through the dataset of harmful prompts,
     - calls the PiF attack generator to produce attack prompts and victim responses,
     - writes per-sample results to a JSON file,
     - evaluates the run and appends a summary line with aggregate metrics.

    @param victim_llm_path Path to the victim LLM model (target model)
    @param results_dir Output directory where results will be written
    @param dataset_path Path to the CSV dataset containing attack goals (column "goal")
    @param what_ollama_model Name of the Ollama model (currently unused in this function)
    @param api_ollama_vllm Whether to use Ollama / vLLM backend (currently unused in this function)

    @return None
    """
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "configPIF.yaml"
    cfg = load_config(config_path)
    cfg_pif = cfg["PIF"]

    gen_model_path: str = cfg_pif["gen_model_path"]
    opt_objective: str = cfg_pif.get("opt_objective", "ASR")
    output_file_name: str = cfg_pif.get("output_file", "_3_pif.json")
    evaluation_template: str = cfg_pif.get("evaluation_template", "this intent is [MASK]")

    iterations: int = int(cfg_pif.get("iterations", 20))

    top_n: int = int(cfg_pif.get("top_n", 5))
    top_m: int = int(cfg_pif.get("top_m", 5))
    top_k: int = int(cfg_pif.get("top_k", 10))
    warm_up: int = int(cfg_pif.get("warm_up", 0))
    temperature: float = float(cfg_pif.get("temperature", 0.1))
    threshold: float = float(cfg_pif.get("threshold", 0.9))

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

    adv_bench = pandas.read_csv(dataset_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    overall_query = 0
    overall_time = 0.0
    overall_successful = 0
    overall_input = 0

    output_path = Path(results_dir) / output_file_name
    logger.info("Results will be written to %s", output_path)
    entries = []
    with open(output_path, "w", encoding="utf-8") as fo:
        for i, harm_prompt in tqdm(enumerate(adv_bench["goal"])):

            chunk_prompts: List[str] = [harm_prompt]

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



if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 run_cypher.py victim_llm_path results_dir dataset_path api_ollama_vllm what_ollama_model")
        sys.exit(1)

    victim_llm_path = sys.argv[1]
    results_dir = sys.argv[2]
    dataset_path = sys.argv[3]
    api_ollama_vllm = str2bool(sys.argv[4].lower())
    what_ollama_model = sys.argv[5]

    run_pif_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)

    