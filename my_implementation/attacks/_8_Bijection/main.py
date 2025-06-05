# main.py

import os
import yaml
import json
import sys

import pandas as pd
from tqdm import tqdm

# Náš vlastní útok a LLM wrapper
from attacks._8_Bijection.bijection_digit_attack import DigitAttack
from attacks._8_Bijection.llm import LLM
from attacks.helpers import load_config
from defense.defense_EA import DefenseEA

def run_digit_attack(run_defense: bool = False):

    defense = DefenseEA()
    # 1) Najdu a načtu configDigit.yaml
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir,"configBijection.yaml")
    if not os.path.isfile(config_path):
        sys.exit(f"Konfigurační soubor nenalezen: {config_path}")

    cfg = load_config(config_path)
    cfg_digit = cfg["Digit"]

    # 2) Rozbalím proměnné z configu
    victim_llm_path = cfg_digit["victim_llm"]
    data_path = cfg_digit["data_path"]
    output_dir = cfg_digit["output_dir"]
    temperature = cfg_digit["temperature"]
    max_token = cfg_digit["max_token"]
    begin = cfg_digit["begin"]
    end = cfg_digit["end"]

    print(f"[INFO] Spouštím DigitAttack s victim_llm={victim_llm_path}, data_path={data_path}")

    # 3) Zkontroluji, že CSV existuje
    if not os.path.isfile(data_path):
        sys.exit(f"Soubor s daty nenalezen: {data_path}")

    # 4) Načtu flip.csv pomocí Pandas
    df = pd.read_csv(data_path)

    # Ověřím, že sloupec 'goal' existuje
    if "goal" not in df.columns:
        sys.exit(f"Chyba: Ve CSV neexistuje sloupec 'goal'.")

    # Oříznu rozsah řádků [begin:end]
    total_rows = len(df)
    if end is None or end > total_rows:
        end = total_rows
    df = df.iloc[begin:end].reset_index(drop=True)

    # 5) Připravím výstupní adresář a JSONL soubor
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "digit_results.jsonl")

    # 6) Inicializuji LLM wrapper
    victim_llm = LLM(
        model_path=victim_llm_path,
        temperature=temperature,
        max_tokens=max_token
    )

    # 7) Vlastní loop: pro každý řádek z CSV
    with open(output_file, "w", encoding="utf-8") as fo:
        for idx, row in enumerate(tqdm(df.itertuples(index=False), desc="Zpracování CSV"), start=begin):
            original_prompt = str(row.goal)  # beru text z 'goal' sloupce

            # 7.1) Vytvořím instanci útoku: každá instance má vlastní náhodnou bijekci
            attack_model = DigitAttack()

            # 7.2) Zakóduji prompt
            gen_info = attack_model.generate(original_prompt)
            encoded_prompt = gen_info["encoded_prompt"]

            # 7.3) Připravím “konverzační“ seznam zpráv (pouze user → encoded_prompt)
            messages = [
                {"role": "user", "content": encoded_prompt}
            ]

            # 7.4) Odešlu do victim LLM
            try:
                if run_defense:
                    messages[-1]['content'] = defense(messages[-1]['content'])
                llm_reply = victim_llm.response(messages)
            except Exception as e:
                llm_reply = f"Chyba při volání LLM: {e}"

            # 7.5) Uložím jeden JSON řádek do výsledků (JSONL)
            entry = {
                "id": idx,
                "original_prompt": gen_info["original_prompt"],
                "encoded_prompt": gen_info["encoded_prompt"],
                "model_reply": llm_reply
            }
            fo.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fo.flush()

    print(f"[INFO] Výsledky jsou uloženy v {output_file}")

# if __name__ == "__main__":
    # run_digit_attack()
