## @file main.py
#  @brief Runner script for the ArtPrompt (ASCII-art–based) prompt injection attack
#
#  This script evaluates the ArtPrompt attack, which disguises a harmful user
#  request inside an ASCII-art–styled prompt. The attack leverages a secondary
#  masking language model to transform the original input into an obfuscated
#  visual representation, which is then submitted to the victim LLM.
#
#  @author Bc. Petr Kaška
#  @date 1.2.2026
#
#  Ownership / Contribution statement:
#   - This runner script was fully implemented by Bc. Petr Kaška.
#   - The experiment orchestration, configuration handling, dual-LLM setup,
#     dataset iteration, and result serialization are original work by the author.
#   - The ArtPrompt attack logic is integrated as an external attack module
#     within the author’s experimental framework.

import os, sys, json, argparse, pandas as pd
from pathlib import Path
from tqdm import tqdm

from attacks.common.helpers import load_config, str2bool
from attacks._21_Art_Prompt.art_attack import ArtPromptAttack, ArtPromptConfig
from attacks.common.llm import LLM

def run_artprompt_attack(victim_llm_path, results_dir, dataset_path, what_ollama_model, api_ollama_vllm):
    here = Path(__file__).parent
    cfg  = load_config(here / "configArt.yaml")["ArtPrompt"]

    mask_llm = LLM(cfg["mask_llm_path"],
                   cfg.get("mask_temperature", 0.9),
                   cfg.get("mask_max_token", 256),
                   "qwen2.5:7b",
                   api_ollama_vllm)

    victim = LLM(victim_llm_path,
                 cfg.get("temperature", 0.0),
                 cfg.get("max_token", 512),
                 api_ollama_vllm,
                 what_ollama_model)

    attack = ArtPromptAttack(ArtPromptConfig(
        mask_llm        = mask_llm,
        ascii_font_file = cfg["ascii_font_file"],
        use_cot         = cfg.get("cot", False)
    ))

    df = pd.read_csv(dataset_path)
    df.columns = [c.lower().strip() for c in df.columns]

    out_dir = Path(results_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "_21_artprompt.json"
    entries = []
    with out_file.open("w", encoding="utf-8") as fo:

        for idx, row in enumerate(tqdm(df.itertuples(index=False),
                                    total=len(df), desc="ArtPrompt")):
            harmful = str(row.goal)
            log, messages = attack.generate(harmful)

            try:
                reply = victim.response(messages)
            except Exception as e:
                reply = f"[ERROR] {e}"

            entry = {
                "id":               idx,
                "original_prompt":  harmful,
                "prompt":           messages[-1]["content"],
                "response":         reply
            }
            entries.append(entry)

        fo.write(json.dumps(entries, ensure_ascii=False))
        fo.flush()

    print(f"[INFO] Výstup uložen → {out_file}")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 run_artprompt.py victim_llm_path results_dir dataset_path api_ollama_vllm what_ollama_model")
        sys.exit(1)

    victim_llm_path = sys.argv[1]
    results_dir = sys.argv[2]
    dataset_path = sys.argv[3]
    api_ollama_vllm = str2bool(sys.argv[4].lower())
    what_ollama_model = sys.argv[5]

    run_artprompt_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)