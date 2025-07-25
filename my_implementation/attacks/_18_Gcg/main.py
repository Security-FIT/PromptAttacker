# TENTO FILE JE MUUUUUUUUUUJ – spouštěč GCG útoku
import os, json, pandas
from tqdm import tqdm

from attacks._18_Gcg.gcg_attack import GCGAttack
from attacks.common.llm import LLM               # Re‑use existing wrapper
from attacks.helpers import load_config
from defense.defense_EA import DefenseEA


def run_gcg_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(script_dir, "configGCG.yaml")
    cfg = load_config(cfg_path)["GCG"]

    # llm_path     = cfg["model_path"]
    # data_path    = cfg["data_path"]
    # out_dir      = cfg["output_dict"]

    begin, end   = cfg.get("begin", 0), cfg.get("end", 519)
    temperature  = cfg["temperature"]
    max_tokens   = cfg["max_tokens"] 

    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, "_18_gcg.jsonl")

    print(f"[INFO] Starting GCG: model={victim_llm_path}, range=[{begin},{end}))")

    victim = LLM(victim_llm_path, temperature, max_tokens, what_ollama_model, api_ollama_vllm)
    attacker = GCGAttack(victim, cfg)
    defense = DefenseEA()

    data = pandas.read_csv(dataset_path)

    with open(out_file, "w", encoding="utf-8") as fo:
        for idx, harmful in tqdm(list(enumerate(data["goal"][begin:end])), total=end - begin):
            log, msgs = attacker.generate(harmful)
            # if run_defense:
                # msgs[-1]["content"] = defense(msgs[-1]["content"])
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

