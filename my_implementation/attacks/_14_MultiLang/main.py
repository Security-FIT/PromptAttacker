## @file main.py
#  @brief Runner script for multilingual prompt-based jailbreak evaluation
#
#  This script evaluates a multilingual jailbreak / prompt-transfer attack
#  where the original adversarial goal is translated into a different language
#  and directly submitted to the victim LLM. The underlying assumption is that
#  safety alignment may be weaker in non-English or cross-lingual settings.
#
#  The script:
#   - Loads experiment parameters from `configMultiLang.yaml`
#   - Reads a CSV dataset containing the original goal and its translation
#     (`goal`, `translation_of_goal`)
#   - Queries the victim LLM using the translated prompt
#   - Stores original prompts, translated prompts, and model responses
#     in JSON format
#
#  @author Bc. Petr Kaška
#  @date 1.2.2026
#
#  Ownership / Contribution statement:
#   - This runner script was fully implemented by Bc. Petr Kaška.
#   - The experiment orchestration, dataset slicing (begin/end),
#     LLM querying logic, and result serialization are original work
#     by the author.
#   - The script integrates a custom experimental setup rather than
#     reusing reference implementations from prior work.
#
#  Research basis:
#   - The experimental idea is inspired by multilingual jailbreak and
#     cross-lingual transfer attacks described in:
#       "Lost in Translation: On the Robustness of Multilingual LLMs
#        to Jailbreak Attacks"
#       arXiv:2401.16765
#       Authors: Kai-Wei Chang, Zhuosheng Zhang, Haoran Shi, Pengfei Liu
#       Initially submitted on arXiv: 30 Jan 2024
#       https://arxiv.org/abs/2401.16765


import os, json, sys,pandas as pd
from tqdm import tqdm
from attacks.common.llm import LLM
from attacks.common.helpers import load_config, str2bool


def run_Multilang_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    cfg_path    = os.path.join(script_dir, "configMultiLang.yaml")
    cfg         = load_config(cfg_path)["MultiLang"]
    print("[CONFIG]", cfg)

    temperature = cfg.get("temperature", 0.0)
    max_token   = cfg.get("max_token", 512)
    begin       = cfg.get("begin", 0)
    end         = cfg.get("end", None)

    victim_llm = LLM(
        model_path=victim_llm_path,
        temperature=temperature,
        max_tokens=max_token,
        ollama_model=what_ollama_model,
        use_ollama=api_ollama_vllm,
    )
    print("[INFO] Initialized LLM client")

    df = pd.read_csv(dataset_path)
    if not {"goal", "translation_of_goal"} <= set(df.columns):
        raise ValueError("CSV musí obsahovat sloupce goal, translation_of_goal")

    if end is None or end > len(df):
        end = len(df)

    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, "_14_multilang.json")
    entries = []
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


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 run_multilang.py victim_llm_path results_dir dataset_path api_ollama_vllm what_ollama_model")
        sys.exit(1)

    victim_llm_path = sys.argv[1]
    results_dir = sys.argv[2]
    dataset_path = sys.argv[3]
    api_ollama_vllm = str2bool(sys.argv[4].lower())
    what_ollama_model = sys.argv[5]

    run_Multilang_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)