# File      : main_overload.py
# Spouštěcí skript pro OverloadAttack (dataset má sloupec „Goal“).

import os
import json
import pandas as pd
from tqdm import tqdm
from dataclasses import fields 

from attacks._17_Overload.overload_attack import OverloadAttacker, OverloadAttackerConfig
from attacks.helpers import load_config
from attacks._17_Overload.llm import LLM
from defense.defense_EA import DefenseEA


def run_overload_attack(run_defense: bool = False) -> None:
    """Spustí OverloadAttack podle YAML konfigurace."""
    defense = DefenseEA() if run_defense else None

    # --- Načtení konfigurace ------------------------------------------------ #
    cfg = load_config(os.path.join(os.path.dirname(__file__), "configOverload.yaml"))
    cfg_ol = cfg["Overload"]

    # --- LLM ---------------------------------------------------------------- #
    victim_llm = LLM(
        model_path=cfg_ol["victim_llm"],
        temperature=cfg_ol.get("temperature", 0.0),
        max_tokens=cfg_ol.get("max_token", 512),
    )

    # --- Dataset ------------------------------------------------------------ #
    data_file = cfg_ol["data_path"]
    df = pd.read_csv(data_file, index_col=0)          # bezejmenný první sloupec = index
    if "Goal" not in df.columns:
        raise KeyError(f"Column 'Goal' not found in {data_file}. "
                       f"Current columns: {list(df.columns)}")
    prompts = df["Goal"].dropna().tolist()

    # --- Výstup ------------------------------------------------------------- #
    out_dir  = cfg_ol["output_dict"]
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "_17_overload.jsonl")  # JSON Lines
    attacker_field_names = {f.name for f in fields(OverloadAttackerConfig)}

    # ↓ odfiltrujeme vše, co dataclass nezná
    attacker_kwargs = {k: v for k, v in cfg_ol.items() if k in attacker_field_names}
    # --- Útočník ------------------------------------------------------------ #
    attacker_cfg = OverloadAttackerConfig(**attacker_kwargs)
    attacker     = OverloadAttacker(attacker_cfg)

    # --- Hlavní smyčka ------------------------------------------------------ #
    with open(out_file, "w", encoding="utf-8") as fo:
        for idx, prompt in tqdm(enumerate(prompts), total=len(prompts)):
            attack_messages = attacker.attack([{"role": "user", "content": prompt}])

            if defense:
                attack_messages[-1]["content"] = defense(attack_messages[-1]["content"])

            llm_resp = victim_llm.response(attack_messages[-1]["content"] )

            print()
            print()
            print()
            print()
            print()
            print(attack_messages)
            print()
            print()
            print()
            print()
            print()
            print()
            print()
            print()
            print()
            print()
            print()
            print()
            print()
            print(llm_resp)
            print()
            print()
            print()
            print()
            print()
            print()
            print()


            json.dump(
                {
                    "id": idx,
                    "prompt_original": prompt,
                    "prompt_attack": attack_messages[-1]["content"],
                    "response": llm_resp,
                },
                fo,
                ensure_ascii=False,
            )
            fo.write("\n")        # JSON na samostatný řádek
            fo.flush()

    print(f"[INFO] Results saved to {out_file}")