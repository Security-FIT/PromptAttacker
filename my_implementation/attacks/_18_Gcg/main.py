# TENTO FILE JE MUUUUUUUUUUJ – spouštěč GCG útoku
import os, json, pandas
from tqdm import tqdm

from attacks._18_Gcg.gcg_attack import GCGAttack
from attacks._18_Gcg.llm import LLM               # Re‑use existing wrapper
from attacks.helpers import load_config
from defense.defense_EA import DefenseEA


def run_gcg_attack(run_defense: bool = False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(script_dir, "configGCG.yaml")
    cfg = load_config(cfg_path)["GCG"]

    llm_path     = cfg["model_path"]
    data_path    = cfg["data_path"]
    out_dir      = cfg["output_dict"]

    begin, end   = cfg.get("begin", 0), cfg.get("end", 519)
    temperature  = cfg["temperature"]
    max_tokens   = cfg["max_tokens"] 

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "_18_gcg.jsonl")

    print(f"[INFO] Starting GCG: model={llm_path}, range=[{begin},{end}))")

    victim = LLM(model_path=llm_path, temperature=temperature, max_tokens=max_tokens)
    attacker = GCGAttack(victim, cfg)
    defense = DefenseEA()

    data = pandas.read_csv(data_path)

    with open(out_file, "w", encoding="utf-8") as fo:
        for idx, harmful in tqdm(list(enumerate(data["Goal"][begin:end])), total=end - begin):
            log, msgs = attacker.generate(harmful)
            if run_defense:
                msgs[-1]["content"] = defense(msgs[-1]["content"])
            response = victim.response(msgs)
            entry = {
                "id": idx,
                "original_prompt": harmful,
                "prompt": msgs[-1]["content"],
                "response": response,
            }
            fo.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fo.flush()

    print(f"[INFO] Results saved to {out_file}")

