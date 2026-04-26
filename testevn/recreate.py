import os
import glob
import json
import shutil
import pandas as pd

# =========================
# NASTAVENÍ
# =========================
DATA_FOLDER = "."
JSON_PATH = "_10_randomsearch.json"

CSV_FILES = [
    os.path.join(DATA_FOLDER, f"out_{i}_harmful_fix_updated.csv")
    for i in [1, 2, 3, 4, 5]
]

CREATE_BACKUP = True
OVERWRITE = True   # False = vytvoří nové *_patched.csv, True = přepíše původní soubory

# =========================
# POMOCNÉ FUNKCE
# =========================
def normalize_columns(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df

import json

def load_json_mapping(json_path):
    def build_mapping(items):
        mapping = {}
        skipped_no_id = 0

        for item in items:
            if not isinstance(item, dict):
                continue
            if "id" not in item:
                skipped_no_id += 1
                continue

            idx = item["id"]
            mapping[idx] = {
                "original_prompt": item.get("original_prompt", None),
                "target_model_answer": item.get("response", None),
            }

        print(f"Načteno validních objektů: {len(mapping)}")
        if skipped_no_id > 0:
            print(f"Přeskočeno objektů bez 'id': {skipped_no_id}")
        return mapping

    # 1) nejdřív zkus normální JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Soubor {json_path} byl načten jako validní JSON.")
        return build_mapping(data)

    except json.JSONDecodeError as e:
        print(f"⚠️ JSON je poškozený, přecházím na best-effort recovery.")
        print(f"   Chyba: {e}")
        print(f"   Řádek: {e.lineno}, sloupec: {e.colno}, pozice: {e.pos}")

    # 2) fallback: best-effort extrakce jednotlivých objektů z JSON pole
    with open(json_path, "r", encoding="utf-8") as f:
        text = f.read()

    decoder = json.JSONDecoder()
    items = []
    pos = 0
    recovered = 0
    skipped_chunks = 0

    while True:
        # najdi začátek dalšího objektu
        start = text.find("{", pos)
        if start == -1:
            break

        try:
            obj, end = decoder.raw_decode(text, start)
            items.append(obj)
            recovered += 1
            pos = end
        except json.JSONDecodeError:
            # nepovedlo se rozparsovat objekt od této závorky,
            # zkus další závorku dál
            skipped_chunks += 1
            pos = start + 1

    print(f"Best-effort recovery: nalezeno {recovered} objektů.")
    print(f"Přeskočeno poškozených chunků: {skipped_chunks}")

    if not items:
        raise ValueError(
            f"Ze souboru {json_path} se nepodařilo načíst žádný validní objekt."
        )

    return build_mapping(items)

def make_output_path(csv_path, overwrite=False):
    if overwrite:
        return csv_path
    return csv_path.replace(".csv", "_patched.csv")

# =========================
# HLAVNÍ LOGIKA
# =========================
def patch_csv_file(csv_path, json_mapping, overwrite=False, create_backup=False):
    if not os.path.exists(csv_path):
        print(f"⚠️ Soubor neexistuje: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    required_columns = [
        "human_score",
        "original_prompt",
        "target_model_answer",
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"V souboru {csv_path} chybí sloupce: {missing}")

    if create_backup and overwrite:
        backup_path = csv_path.replace(".csv", "_backup.csv")
        shutil.copy(csv_path, backup_path)
        print(f"Záloha vytvořena: {backup_path}")

    df["human_score"] = pd.to_numeric(df["human_score"], errors="coerce")

    replaced_count = 0
    missing_json_count = 0

    for row_idx in df.index:
        human_score = df.at[row_idx, "human_score"]

        if pd.isna(human_score):
            continue

        if human_score in [0, 1]:
            if row_idx not in json_mapping:
                missing_json_count += 1
                continue

            json_item = json_mapping[row_idx]

            if json_item["original_prompt"] is not None:
                df.at[row_idx, "original_prompt"] = json_item["original_prompt"]

            if json_item["target_model_answer"] is not None:
                df.at[row_idx, "target_model_answer"] = json_item["target_model_answer"]

            replaced_count += 1

    output_path = make_output_path(csv_path, overwrite=overwrite)
    df.to_csv(output_path, index=False)

    print(f"\nSoubor: {csv_path}")
    print(f"Uloženo jako: {output_path}")
    print(f"Nahrazeno řádků: {replaced_count}")
    print(f"Chybějící id v JSON: {missing_json_count}")

# =========================
# SPUŠTĚNÍ
# =========================
if not os.path.exists(JSON_PATH):
    raise FileNotFoundError(f"JSON soubor nebyl nalezen: {JSON_PATH}")

json_mapping = load_json_mapping(JSON_PATH)

for csv_file in CSV_FILES:
    patch_csv_file(
        csv_path=csv_file,
        json_mapping=json_mapping,
        overwrite=OVERWRITE,
        create_backup=CREATE_BACKUP
    )

print("\nHotovo.")