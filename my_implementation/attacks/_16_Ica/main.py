# attacks/_3_ICA/main.py
import os, json, pandas as pd
from pathlib import Path
from tqdm import tqdm

from attacks._16_Ica.ica_attack import ICAAttack
from attacks._16_Ica.llm import LLM
from attacks.helpers import load_config
from defense.defense_EA import DefenseEA   

def run_ica_attack(use_defense: bool = False):
    # ------------ načti yaml config ------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(script_dir + "/configIca.yaml")["ICA"]

    victim_llm  = cfg["victim_llm"]
    data_path   = cfg["data_path"]
    out_dir     = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    temperature = cfg.get("temperature", 0.0)
    max_token   = cfg.get("max_token", 512)
    cot         = cfg.get("cot", False)
    begin       = cfg.get("begin", 0)
    end         = cfg.get("end", None)

    print(f"[INFO] ICA-Attack  •  model={victim_llm}  •  COT={cot}")

    # ------------ init LLM & attacker ----------------------------------
    victim   = LLM(victim_llm, temperature, max_token)
    attacker = ICAAttack(cot=cot)
    defense  = DefenseEA() if use_defense else None

    # ------------ načti dataset ----------------------------------------
    df = pd.read_csv(data_path)
    df.columns = [c.lower().strip() for c in df.columns]   # Goal → goal, Target → target …
    if end is None or end > len(df):
        end = len(df)

    # ------------ inference smyčka -------------------------------------
    out_file = out_dir / "_16_ica.json"
    with out_file.open("w", encoding="utf-8") as fo:
        for idx, row in enumerate(
                tqdm(df.iloc[begin:end].itertuples(index=False),
                     total=end - begin, desc="ICA"),
                start=begin):
            harmful_prompt = str(row.goal)
            log, msgs = attacker.generate(harmful_prompt)

            if defense:
                msgs[-1]["content"] = defense(msgs[-1]["content"])

            try:
                reply = victim.response(msgs)
            except Exception as e:
                reply = f"[ERROR] {e}"

            fo.write(json.dumps({
                "id": idx + begin,
                "category": getattr(row, "category", ""),
                "original_prompt": harmful_prompt,
                "prompt": msgs[-1]["content"],
                "response": reply
            }, ensure_ascii=False) + "\n")
            fo.flush()

    print(f"[INFO] Výstup uložen → {out_file}")