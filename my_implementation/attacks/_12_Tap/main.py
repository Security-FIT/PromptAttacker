from __future__ import annotations

import json, os, time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from attacks.helpers import load_config
from defense.defense_EA import DefenseEA
from attacks._12_Tap.attack_tap import (
    TAPAttacker,
    TAPAttackerConfig,
    BaseLLMConfig,
    LLMGenerateConfig,
)

# --------------------------------------------------------------------------- #
def run_tap_attack(run_defense: bool = False) -> None:
    """Spustí TAP útok podle `configTap.yaml`."""
    root = Path(__file__).resolve().parent
    cfg  = load_config(root / "configTap.yaml")["TAP"]

    # --- YAML ----------------------------------------------------------------
    data_path   = cfg["data_path"]
    out_dir     = Path(cfg["output_dir"])
    victim_path = cfg["victim_llm"]

    attacker_name = cfg.get("attacker_llm")
    temperature   = cfg.get("temperature", 1.0)
    max_token     = cfg.get("max_token", 1024)
    width         = cfg.get("width", 10)
    branching     = cfg.get("branching", 10)
    depth         = cfg.get("depth", 10)
    target_str    = cfg.get("target_str", "Sure.")
    begin         = cfg.get("begin", 0)
    end           = cfg.get("end")
    sleep_sec     = cfg.get("sleep", 0.0)
    apply_defense = cfg.get("apply_defense", False)

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
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "_12_tap.jsonl"

    # --- iterace --------------------------------------------------------------
    with out_file.open("w", encoding="utf-8") as fo:
        series = adv_bench["goal"][begin:end]
        for idx, rec in tqdm(series.items()):
            goal = rec["goal"] if isinstance(rec, dict) else rec
            print(f"[INFO] Goal {idx}: {goal[:60]}…")

            # 1) najdeme adversarial prompt
            conv = [{"role": "user", "content": goal}]
            result = tap.attack(messages=conv)
            attack_prompt = result[0]["content"] if result else "[ATTACK FAILED]"

            # 2) volitelná obrana
            if apply_defense and defense:
                attack_prompt = defense(attack_prompt)

            # 3) dotaz na victim – použijeme už existující `tap.target_llm`
            response = tap.target_llm.response(
                [{"role": "user", "content": attack_prompt}]
            )
            print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            print("ZDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")


            entry = {
                "id": idx,
                "original_prompt": goal,
                "prompt": attack_prompt,
                "response": response,
            }
            fo.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fo.flush()
            time.sleep(sleep_sec)

    print(f"[INFO] Results saved to {out_file}")
