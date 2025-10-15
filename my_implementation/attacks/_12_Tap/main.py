from __future__ import annotations

import json, os, time, sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm
from datetime import timedelta

from attacks.helpers import load_config, str2bool
from defense.defense_EA import DefenseEA
from attacks._12_Tap.attack_tap import (
    TAPAttacker,
    TAPAttackerConfig,
    BaseLLMConfig,
    LLMGenerateConfig,
)

# --------------------------------------------------------------------------- #
def run_tap_attack(victim_llm_path, results_dir, dataset_path, what_ollama_model, api_ollama_vllm) -> None:
    """Spustí TAP útok podle `configTap.yaml`."""
    root = Path(__file__).resolve().parent
    cfg  = load_config(root / "configTap.yaml")["TAP"]

    # --- YAML ----------------------------------------------------------------
    data_path   = dataset_path
    out_dir     = Path(cfg["output_dir"])
    victim_path = victim_llm_path

    attacker_name = victim_llm_path
    temperature   = 1
    max_token     = 1024
    width         = 8
    branching     = 3
    depth         = 3
    target_str    = "Sure."
    begin         = 0
    end           = "end"
    sleep_sec     = 0.0
    apply_defense = False

    print("[INFO] ============ TAP Experiment ============")
    print(f"[INFO] victim model : {victim_path}")
    print(f"[INFO] attacker LLM : {attacker_name}")
    print(f"[INFO] width/branch/depth : {width}/{branching}/{depth}")
    print(f"[INFO] defense      : {'ON' if apply_defense else 'OFF'}")

    # --- TAP konfigurace ------------------------------------------------------
    tgt_cfg = BaseLLMConfig(model=victim_path)
    atk_cfg = BaseLLMConfig(model=attacker_name)
    gen_cfg = LLMGenerateConfig(max_tokens=max_token, temperature=temperature)

    tap_cfg = TAPAttackerConfig(
        attacker_cls="TAPAttacker",
        attacker_name="tap_experiment",
        target_str=target_str,
        width=width,
        branching_factor=branching,
        depth=depth,
        target_llm_config=tgt_cfg,
        target_llm_gen_config=gen_cfg,
        attack_llm_config=atk_cfg,
        attack_llm_gen_config=gen_cfg,
    )
    tap = TAPAttacker(config=tap_cfg)

    defense = DefenseEA() if apply_defense else None

    # --- dataset --------------------------------------------------------------
    adv_bench = pd.read_csv(data_path)
    out_dir = Path(results_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "_12_tap.json"
    entries = []
    # --- iterace --------------------------------------------------------------
    with out_file.open("w", encoding="utf-8") as fo:
        series = adv_bench["goal"]
        for idx, rec in tqdm(series.items()):
            goal = rec["goal"] if isinstance(rec, dict) else rec
            print(f"[INFO] Goal {idx}: {goal[:60]}…")

            last_best_prompt = None
            def hook(state: Dict):
                nonlocal last_best_prompt
                bp = state.get("best_prompt")
                if isinstance(bp, str) and bp.strip():
                    last_best_prompt = bp

            # 1) najdeme adversarial prompt
            conv = [{"role": "user", "content": goal}]
            result = tap.attack(messages=conv,
            progress_hook=hook,
            progress_every=10,
            time_budget_sec=120,   # měkký limit → vrátí best-so-far
            max_nodes=500         )
            attack_prompt = result[0]["content"] if result else "[ATTACK FAILED]"

            # 2) volitelná obrana
            if apply_defense and defense:
                attack_prompt = defense(attack_prompt)

            # 3) dotaz na victim – použijeme už existující `tap.target_llm`
            response = tap.target_llm.response(
                [{"role": "user", "content": attack_prompt}]
            )
            # print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            # print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            # print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            # print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            # print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            # print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            # print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            # print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            # print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            # print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            # print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            # print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            # print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            # print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            # print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")


            entry = {
                "id": idx,
                "original_prompt": goal,
                "prompt": attack_prompt,
                "response": response,
            }
            entries.append(entry)
        fo.write(json.dumps(entries, ensure_ascii=False) + "\n")
        fo.flush()
        time.sleep(sleep_sec)

    print(f"[INFO] Results saved to {out_file}")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 run_cypher.py victim_llm_path results_dir dataset_path api_ollama_vllm what_ollama_model")
        sys.exit(1)

    victim_llm_path = sys.argv[1]
    results_dir = sys.argv[2]
    dataset_path = sys.argv[3]
    api_ollama_vllm = str2bool(sys.argv[4].lower())
    what_ollama_model = sys.argv[5]

    run_tap_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)