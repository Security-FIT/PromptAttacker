import os, json, pandas as pd
from tqdm import tqdm
from attacks.common.llm import LLM
from attacks.helpers import load_config
from defense.defense_EA import DefenseEA    # zůstává, i když se zde nepoužije

def run_Multilang_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):
    """Multi‑language útok – LLM dostane FR prompt, do záznamu uloží i EN originál."""

    # ---------- konfigurace -------------------------------------------------
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    cfg_path    = os.path.join(script_dir, "configMultiLang.yaml")
    cfg         = load_config(cfg_path)["MultiLang"]
    print("[CONFIG]", cfg)

    temperature = cfg.get("temperature", 0.0)
    max_token   = cfg.get("max_token", 512)
    begin       = cfg.get("begin", 0)
    end         = cfg.get("end", None)

    # ---------- LLM klient --------------------------------------------------
    victim_llm = LLM(
        model_path=victim_llm_path,
        temperature=temperature,
        max_tokens=max_token,
        ollama_model=what_ollama_model,
        use_ollama=api_ollama_vllm,
    )
    print("[INFO] Initialized LLM client")

    # ---------- načti dataset ----------------------------------------------
    df = pd.read_csv(dataset_path)
    if not {"goal", "translation_of_goal"} <= set(df.columns):
        raise ValueError("CSV musí obsahovat sloupce goal, translation_of_goal")

    if end is None or end > len(df):
        end = len(df)

    # ---------- výstupní soubor --------------------------------------------
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, "_14_multilang.json")
    entries = []
    # ---------- hlavní smyčka ----------------------------------------------
    with open(output_file, "w", encoding="utf-8") as fo:
        for idx, row in tqdm(df.iloc[begin:end].iterrows(), total=end - begin):
            original_goal   = row["goal"]                 
            ch_prompt       = row["translation_of_goal"]  

            chat_prompt = [{"role": "user", "content": ch_prompt}]
            llm_response = victim_llm.response(chat_prompt)

            entry = {
                "id": idx,
                "original_prompt": original_goal,
                "prompt": ch_prompt,
                "response": llm_response,
            }
            entries.append(entry)
        fo.write(json.dumps(entries, ensure_ascii=False) + "\n")
        fo.flush()
    print(f"[INFO] Results saved to {output_file}")
