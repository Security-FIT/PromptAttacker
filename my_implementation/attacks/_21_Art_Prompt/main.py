# attacks/_24_ArtPrompt/main.py
import os, json, argparse, pandas as pd
from pathlib import Path
from tqdm import tqdm

from attacks.helpers import load_config
from attacks._21_Art_Prompt.art_attack import ArtPromptAttack, ArtPromptConfig
from attacks.common.llm import LLM
from defense.defense_EA import DefenseEA


def run_artprompt_attack(victim_llm_path, results_dir, dataset_path, what_ollama_model, api_ollama_vllm):
    here = Path(__file__).parent
    cfg  = load_config(here / "configArt.yaml")["ArtPrompt"]

    mask_llm = LLM(cfg["mask_llm_path"],
                   cfg.get("mask_temperature", 0.7),
                   cfg.get("mask_max_token", 256),
                   "qwen2.5:7b",
                   True)

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
    # defense = DefenseEA() if use_defense else None

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

            # if defense:
                # messages[-1]["content"] = defense(messages[-1]["content"])

            # if not messages[-1]["content"]:
                # continue

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