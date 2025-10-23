#!/usr/bin/env python3
import csv
import json
import os
import sys
from typing import List, Optional

# ---------- Konfigurace (upravit podle potřeby) ----------

JSON_DIR = "/storage/brno2/home/xkaska01/master/my_implementation/results/DATASET_FULL_command-r:35b"  # složka kde jsou JSONy
OUTPUT_JSONL = "selected_examples.jsonl"
OUTPUT_JSON = "selected_examples.json"
# ---------------------------------------------------------

def normalize_attack_name(s: str) -> str:
    """Normalize attack_type for matching (lowercase, replace spaces)."""
    return s.strip().lower().replace(" ", "_")

def json_path_for_attack(attack_type: str, json_dir: str) -> str:
    """
    Najde JSON soubor pro daný attack_type v json_dir.
    Hledá tyto varianty:
      1) exact match: <attack_type>.json
      2) prefixed numeric files: _<num>_<attack_type>.json  (např. _1_cipher.json)
      3) substring match: any file containing attack_type (case-insensitive)
      4) fallback: první .json v adresáři
    """
    norm = normalize_attack_name(attack_type)
    # 1) exact
    cand = os.path.join(json_dir, f"{norm}.json")
    if os.path.exists(cand):
        return cand

    # 2) prefixed with number underscore
    # match files like "_1_cipher.json" or "01_cipher.json" or "1_cipher.json"
    for fn in os.listdir(json_dir):
        if not fn.lower().endswith(".json"):
            continue
        name = fn[:-5]  # without .json
        # split by underscore and check last token(s)
        parts = name.lower().split("_")
        # try to find attack_type at the end
        if len(parts) >= 2 and parts[-1] == norm:
            return os.path.join(json_dir, fn)
        # sometimes format: "_1-cipher" or "1-cipher"
        if "-" in name:
            if name.lower().endswith("-" + norm):
                return os.path.join(json_dir, fn)

    # 3) substring match (case-insensitive)
    for fn in os.listdir(json_dir):
        if not fn.lower().endswith(".json"):
            continue
        if norm in fn.lower():
            return os.path.join(json_dir, fn)

    # 4) fallback first json
    all_jsons = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.lower().endswith(".json")]
    if all_jsons:
        return all_jsons[0]

    raise FileNotFoundError(f"No JSON file found for attack_type='{attack_type}' in {json_dir}")

def parse_scores_row(row: dict) -> List[Optional[int]]:
    """Z CSV řádku vrátí seznam skóre (None pokud chybí)."""
    scores: List[Optional[int]] = []
    for key in row:
        if key.lower() == "attack_type":
            continue
        val = row[key]
        if val is None or val == "":
            scores.append(None)
        else:
            try:
                scores.append(int(float(val)))
            except Exception:
                scores.append(None)
    return scores

def choose_index(scores: List[Optional[int]]) -> Optional[int]:
    """Vybere index: první 10, jinak první 9, jinak index s max hodnotou, jinak None."""
    for i, s in enumerate(scores):
        if s == 10:
            return i
    for i, s in enumerate(scores):
        if s == 9:
            return i
    # fallback: nejlepší hodnota
    best_idx = None
    best_val = -1
    for i, s in enumerate(scores):
        if s is not None and s > best_val:
            best_val = s
            best_idx = i
    return best_idx

def load_json_list(fp: str) -> List[dict]:
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {fp}, got {type(data)}")
    return data

def main(csv_path: str, json_dir: str):
    if not os.path.exists(csv_path):
        print(f"[ERR] CSV not found: {csv_path}", file=sys.stderr); sys.exit(1)
    if not os.path.isdir(json_dir):
        print(f"[ERR] JSON dir not found: {json_dir}", file=sys.stderr); sys.exit(1)

    selected = []
    with open(csv_path, newline="", encoding="utf-8") as cf:
        reader = csv.DictReader(cf)
        for row in reader:
            # pokusit se získat sloupec attack_type (rozdílné názvy HEAD)
            attack_type = row.get("attack_type") or row.get("attack type") or row.get("attack-type")
            if not attack_type:
                print("[WARN] Řádek bez attack_type, přeskočeno.", file=sys.stderr)
                continue
            scores = parse_scores_row(row)
            idx = choose_index(scores)
            try:
                json_fp = json_path_for_attack(attack_type, json_dir)
            except FileNotFoundError as e:
                print(f"[WARN] {e}", file=sys.stderr)
                continue

            try:
                items = load_json_list(json_fp)
            except Exception as e:
                print(f"[ERROR] Nelze načíst JSON {json_fp}: {e}", file=sys.stderr)
                continue

            if idx is None or idx < 0 or idx >= len(items):
                print(f"[WARN] Nevalidní index {idx} pro {attack_type} (soubor má {len(items)} položek). Používám index 0.", file=sys.stderr)
                idx = 0

            item = items[idx]
            # metadata
            item_meta = {
                "_selected_attack_type": attack_type,
                "_selected_index": idx,
                "_selected_score": scores[idx] if (0 <= idx < len(scores)) else None,
                "_source_file": os.path.basename(json_fp),
            }
            out_obj = dict(item_meta)
            out_obj.update(item)
            selected.append(out_obj)

    # save outputs
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as outf:
        for obj in selected:
            outf.write(json.dumps(obj, ensure_ascii=False) + "\n")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as outf:
        json.dump(selected, outf, ensure_ascii=False, indent=2)

    print(f"[OK] Vybráno {len(selected)} příkladů. Uloženo do {OUTPUT_JSONL} a {OUTPUT_JSON}")

if __name__ == "__main__":
    main("/storage/brno2/home/xkaska01/master/my_implementation/evaluate/command-r:35b.csv", JSON_DIR)
