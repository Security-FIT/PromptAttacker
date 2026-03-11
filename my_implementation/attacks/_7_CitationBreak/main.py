## @file main.py
#  @brief Script for running Citation-based prompt injection attacks
#
#  This script loads a dataset containing goals and citation blocks and applies
#  a citation-driven attack strategy against a victim LLM. The attacker constructs
#  prompts that reference external citations and then queries the victim model.
#  Results are stored in JSON format for further evaluation.
#
#  @author Bc. Petr Kaška
#  @date 3.1.2026
import os, json, sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from attacks._7_CitationBreak.citation_attack import CitationAttack
from attacks.common.llm import LLM
from attacks.common.helpers import load_config, str2bool


def run_cite_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configCitation.yaml")
    cfg         = load_config(config_path)
    cp          = cfg["Citation"]

    adv_bench = pd.read_csv(dataset_path, dtype=str, keep_default_na=False)

    required_cols = ["goal", "translation_of_goal", "citation", "target"]
    missing = [c for c in required_cols if c not in adv_bench.columns]
    if missing:
        raise ValueError(f"Dataset musí obsahovat sloupce: {required_cols}. Chybí: {missing}")

    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, "_7_citation.json")

    victim = LLM(
        victim_llm_path,
        cp.get("temperature", 0.0),
        cp.get("max_token", 512),
        what_ollama_model,
        api_ollama_vllm
    )
    attacker = CitationAttack()
    entries = []
    with open(output_file, "w", encoding="utf-8") as fo:
        for i, row in enumerate(tqdm(adv_bench.itertuples(index=False), total=len(adv_bench))):
            goal      = (row.goal or "").strip()
            target    = (row.target or "").strip()
            citation  = (row.citation or "").strip()

            if citation:
                harm_prompt = f"{goal}\n\This Prompt is based on citation: {citation}"
                citation_block = citation  

            messages = attacker.generate(
                key=target,
                goal=harm_prompt,     
                citation_block=citation_block,
                style="citation"
            )

            try:
                reply = victim.response(messages)
            except Exception as e:
                reply = f"[ERROR] {e}"

            entries.append({
                "id": i,
                "original_prompt": goal,
                "prompt":        messages[-1]["content"],
                "response":      reply
            })
        fo.write(json.dumps(entries, ensure_ascii=False) + "\n")
        fo.flush()

    print(f"[INFO] Výstup uložen → {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 run_cypher.py victim_llm_path results_dir dataset_path api_ollama_vllm what_ollama_model")
        sys.exit(1)

    victim_llm_path = sys.argv[1]
    results_dir = sys.argv[2]
    dataset_path = sys.argv[3]
    api_ollama_vllm = str2bool(sys.argv[4].lower())
    what_ollama_model = sys.argv[5]

    run_cite_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)