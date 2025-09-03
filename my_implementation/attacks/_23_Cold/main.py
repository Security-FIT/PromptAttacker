import os
import json
import yaml
import argparse
from types import SimpleNamespace

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attacks._23_Cold.attack_suffix import attack_generation


def load_config(path):
    """Thin wrapper around yaml.safe_load for parity with other attacks."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(model_id: str, dtype="auto"):
    """Load HF model & tokenizer on the first available GPU/CPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype if dtype != "auto" else None
    ).to(device)
    model.eval()
    return model, tokenizer, device


def dataframe_to_jsonl(df: pd.DataFrame, file_path: str):
    """Save the attack dataframe in the same jsonl format as other attacks."""
    entries = []
    with open(file_path, "w", encoding="utf-8") as fo:
        for idx, row in df.iterrows():
            record = {
                "id": int(idx),
                "original_prompt": str(row["prompt"]).strip(),
                "prompt": str(row.get("prompt_with_adv", "")),
                "response": str(row.get("output", "")),
                # Keep the raw suffix for possible analysis
                # "adv_suffix": str(row.get("adv", ""))
            }
            entries.append(record)
        fo.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_cold_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):
    # ---------------------------------------------------------------- config --
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configCold.yaml")
    cfg         = load_config(config_path)
    cfgCold      = cfg["Cold"]

    # victim_llm_path = cfgCold["victim_llm"]
    # dataset_path = cfgCold["data_path"]
    # results_dir = cfgCold["output_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # fallbacks + defaults ----------------------------------------------------
    begin = cfgCold.get("begin", 0)
    end = cfgCold.get("end", None)

    # ----------------------------------------------------- model & tokenizer --
    print(f"[INFO] Loading model '{victim_llm_path}'...")
    model, tokenizer, device = build_model(victim_llm_path)

    # ----------------------------------------------------------- build args --
    # Anything inside ColdAttack apart from meta‑keys goes into the args bag
    # so that attack_suffix.attack_generation can access it as attributes.
    meta_keys = {"victim_llm", "dataset_path", "output_dir"}
    args_dict = {k: v for k, v in cfgCold.items() if k not in meta_keys}
    args_dict.setdefault("start", begin)
    args_dict.setdefault("end", end if end is not None else 10**9)
    args = SimpleNamespace(**args_dict)


    # ------------------------------------------------------------ run attack --
    print("[INFO] Running cold attack...")
    df_results = attack_generation(model, tokenizer, device, args, dataset_path)

    # The reference implementation of cold attack does not return anything.
    # If the user hasn't patched attack_suffix.attack_generation to return a
    # dataframe, we fall back to reading a temp parquet if they saved it there.
    if df_results is None:
        raise RuntimeError(
            "attack_generation returned 'None'. "
            "Please add 'return results' as the last line of the function "
            "or otherwise expose the dataframe."
        )

    # ------------------------------------------------------------- save jsonl --
    output_file = os.path.join(results_dir, "_23_cold.json")
    dataframe_to_jsonl(df_results, output_file)
    print(f"[INFO] Results saved to {output_file}")
