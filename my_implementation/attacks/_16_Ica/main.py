# attacks/_3_ICA/main.py
import os, sys,json, pandas as pd
from pathlib import Path
from tqdm import tqdm

from attacks._16_Ica.ica_attack import ICAAttack
from attacks.common.llm import LLM
from attacks.helpers import load_config, str2bool
from defense.defense_EA import DefenseEA   

def run_ica_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):
    # ------------ načti yaml config ------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(script_dir + "/configIca.yaml")["ICA"]

    # victim_llm  = cfg["victim_llm"]
    # data_path   = cfg["data_path"]
    # out_dir     = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    temperature = cfg.get("temperature", 0.0)
    max_token   = cfg.get("max_token", 512)
    cot         = cfg.get("cot", False)
    begin       = cfg.get("begin", 0)
    end         = cfg.get("end", None)

    print(f"[INFO] ICA-Attack  •  model={victim_llm_path}  •  COT={cot}")

    # ------------ init LLM & attacker ----------------------------------
    victim   = LLM(victim_llm_path, temperature, max_token, what_ollama_model, api_ollama_vllm)
    attacker = ICAAttack(cot=cot)
    # defense  = DefenseEA() if use_defense else None

    # ------------ načti dataset ----------------------------------------
    df = pd.read_csv(dataset_path)

    # ------------ inference smyčka -------------------------------------

    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, '_16_ica.json')
    print(f"[INFO] Output file: {out_file}")
    print()
    print()
    print()
    print()

    entries = []
    with open(out_file, "w", encoding="utf-8") as fo:
        for idx, row in tqdm(
            enumerate(df["goal"][begin:end]), total=end - begin
        ):
            # harmful_prompt = str(row.goal)
            log, msgs = attacker.generate(row)

            # if defense:
            #     msgs[-1]["content"] = defense(msgs[-1]["content"])

            try:
                reply = victim.response(msgs)
            except Exception as e:
                reply = f"[ERROR] {e}"

            entry = {
                "id": idx + begin,
                "original_prompt": row,
                "prompt": msgs[-1]["content"],
                "response": reply
            }
            entries.append(entry)
        fo.write(json.dumps(entries, ensure_ascii=False) + "\n")
        fo.flush()

    print(f"[INFO] Výstup uložen → {out_file}")




if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 run_cypher.py victim_llm_path results_dir dataset_path api_ollama_vllm what_ollama_model")
        sys.exit(1)

    victim_llm_path = sys.argv[1]
    results_dir = sys.argv[2]
    dataset_path = sys.argv[3]
    api_ollama_vllm = str2bool(sys.argv[4].lower())
    what_ollama_model = sys.argv[5]

    run_ica_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)