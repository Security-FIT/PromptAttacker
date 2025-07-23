import os
import yaml
import json
import sys

import pandas as pd
from tqdm import tqdm

# Náš vlastní útok a LLM wrapper
from attacks._8_Bijection.bijection_digit_attack import DigitAttack
from attacks.common.llm import LLM
from attacks.helpers import load_config
from defense.defense_EA import DefenseEA

def run_bijection_attack(run_defense: bool = False):
    defense = DefenseEA()

    # 1) Najdu a načtu configBijection.yaml
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configBijection.yaml")
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
    begin = cfg_digit.get("begin", 0)
    end = cfg_digit.get("end", None)
    kt = cfg_digit.get("kt", 1)
    kp = cfg_digit.get("kp", 1)

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
    output_file = os.path.join(output_dir, "_8_bijection_results.jsonl")

    # 6) Inicializuji LLM wrapper
    victim_llm = LLM(
        model_path=victim_llm_path,
        temperature=temperature,
        max_tokens=max_token
    )

    with open(output_file, "w", encoding="utf-8") as fo:
        for idx, row in enumerate(tqdm(df.itertuples(index=False), desc="Zpracování CSV"), start=begin):
            original_prompt = str(row.goal).strip()
            if not original_prompt:
                continue

            # --- START OPRAVENÉ ČÁSTI ---

            # 1) Vytvořím instanci útoku (A.1–A.3)
            attack_model = DigitAttack(kt=kt, kp=kp)

            # 2) Vygeneruji seznam zpráv (system, teaching, practice, a zakončený zakódovaným promptem)
            messages = attack_model.generate(original_prompt)

            # 3) Odešlu `messages` do LLM
            try:
                llm_reply = victim_llm.response(messages)
            except Exception as e:
                llm_reply = f"Chyba při volání LLM: {e}"

            # 4) Zakódovaný prompt je v poslední položce `messages`
            encoded_prompt = messages[-1]["content"]

            # --- KONEC OPRAVENÉ ČÁSTI ---

            # 5) Složím JSON objekt a zapíšu ho jako jeden řádek JSONL
            entry = {
                "id": idx,
                "original_prompt": original_prompt,
                "encoded_prompt": encoded_prompt,
                "model_reply": llm_reply
            }
            fo.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fo.flush()

    print(f"[INFO] Výsledky jsou uloženy v {output_file}")
