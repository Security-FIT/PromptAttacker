# attacks/_9_Citation/main.py
import os, json, argparse, time, random, tqdm, yaml, pandas as pd
from pathlib import Path

from attacks._7_CitationBreak.citation_attack import CitationAttack, get_attack_prompt
from attacks.common.llm import LLM
from attacks.helpers import load_config         
from defense.defense_EA import DefenseEA        


META = {"topic", "key", "cata"}          # sloupce, které ignorujeme

def run_cite_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):
    defense = DefenseEA()

    # načtení configu
    script_dir  = Path(__file__).parent
    cfg         = load_config(script_dir / "configCitation.yaml")
    cp          = cfg["Citation"]

    # dataset
    rows        = [json.loads(l) for l in Path(cp["data_path"]).open()]
    out_file    = Path(results_dir) / "_7_citation.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # LLM + attacker
    victim   = LLM(victim_llm_path, cp.get("temperature",0.0), cp.get("max_token",512), what_ollama_model,
        api_ollama_vllm)
    attacker = CitationAttack()

    with out_file.open("w", encoding="utf-8") as fo:
        for idx, row in enumerate(tqdm.tqdm(rows, desc="CitationAttack")):
            goal = row["topic"]
            key  = row.get("key", goal)

            # nasbíráme *všechny* citace v řádku
            citation_items = []
            # nejdřív nested "citation"
            if "citation" in row and row["citation"].get("content"):
                citation_items.append((row["citation"]["type"], row["citation"]["content"]))

            # pak ostatní sloupce
            for name, val in row.items():
                if name in META or not val or name == "citation":
                    continue
                citation_items.append((name, val))

            for cit_type, raw_block in citation_items:
                cit_block = f"1. {raw_block}"
                messages  = attacker.generate(key, goal, cit_block, style=cit_type)

                # if run_defense:
                    # messages[-1]["content"] = defense(messages[-1]["content"])

                try:
                    reply = victim.response(messages)
                except Exception as e:
                    reply = f"[ERROR] {e}"

                fo.write(json.dumps({
                    "id": idx,
                    "citation_type": cit_type,
                    "prompt":        messages[-1]["content"],
                    "response":      reply
                }, ensure_ascii=False) + "\n")
                fo.flush()

    print(f"[INFO] Výstup uložen → {out_file}")