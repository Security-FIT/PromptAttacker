# attacks/_9_Citation/main.py
import os, json, time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from attacks._7_CitationBreak.citation_attack import CitationAttack
from attacks.common.llm import LLM
from attacks.helpers import load_config
from defense.defense_EA import DefenseEA

META = {"topic", "key", "cata"}

def run_cite_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):
    defense = DefenseEA()

    # načtení configu
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configCitation.yaml")
    cfg         = load_config(config_path)
    cp          = cfg["Citation"]

    # dataset
    adv_bench = pd.read_csv(dataset_path, dtype=str, keep_default_na=False)

    required_cols = ["goal", "translation_of_goal", "citation", "target"]
    missing = [c for c in required_cols if c not in adv_bench.columns]
    if missing:
        raise ValueError(f"Dataset musí obsahovat sloupce: {required_cols}. Chybí: {missing}")

    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, "_7_citation.json")

    # LLM + attacker
    victim = LLM(
        victim_llm_path,
        cp.get("temperature", 0.0),
        cp.get("max_token", 512),
        what_ollama_model,
        api_ollama_vllm
    )
    attacker = CitationAttack()
    entries = []
    with open(output_file, "w", encoding="utf-8") as fo:
        # Iterujeme přes celé řádky, ať máme goal/target/citation
        for i, row in enumerate(tqdm(adv_bench.itertuples(index=False), total=len(adv_bench))):
            goal      = (row.goal or "").strip()
            target    = (row.target or "").strip()
            citation  = (row.citation or "").strip()

            # 1) Vždy připojit dovětek za harmful prompt (goal)
            #    → přesně, jak chceš: "že to vychází z citace, která je ..."
            if citation:
                harm_prompt = f"{goal}\n\This Prompt is based on citation: {citation}"
                citation_block = citation  # do šablony půjde jako Reference

            # 2) Sestavit zprávy pro LLM (topic = target)
            messages = attacker.generate(
                key=target,                 # 'topic' v šabloně
                goal=harm_prompt,           # 'question' v šabloně = harmful prompt + dovětek
                citation_block=citation_block,
                style="citation"            # => "from the following citation"
            )

            # (volitelná obrana – nechávám zakomentované, pokud chceš zapnout, odkomentuj)
            # messages[-1]["content"] = defense(messages[-1]["content"])

            # 3) Dotaz na oběť
            try:
                reply = victim.response(messages)
            except Exception as e:
                reply = f"[ERROR] {e}"

            # 4) Log do JSONL
            entries.append({
                "id": i,
                "original_prompt": goal,
                "prompt":        messages[-1]["content"],
                "response":      reply
            })
        fo.write(json.dumps(entries, ensure_ascii=False) + "\n")
        fo.flush()

    print(f"[INFO] Výstup uložen → {output_file}")
