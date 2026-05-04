#!/usr/bin/env python3
"""Run inference for one prepared attack JSON file.

Input files are expected to be JSON arrays produced by the individual attack
methods, usually with at least these fields:

    id, original_prompt, prompt

For every item the script sends `prompt` to the selected victim model and writes
the enriched result to `output_dir/<input-file-name>` with the normalized schema:

    id, original_prompt, prompt, response

The backend is controlled by the `api_ollama_vllm` CLI argument:

    true  -> use Ollama HTTP API with `what_ollama_model`
    false -> use local vLLM with `victim_llm_path`
"""

import sys
import os
import json
from tqdm import tqdm
from pathlib import Path
import time

from attacks.common.llm import LLM
from attacks.common.helpers import str2bool


def ensure_dir(d: str) -> None:
    """@brief Create an output directory if it does not exist.

    @param d Directory path.
    """
    os.makedirs(d, exist_ok=True)


def read_json(path: str):
    """@brief Read a JSON file and repair missing/empty/invalid files.

    If the file does not exist, is empty, or contains invalid JSON, the function
    writes an empty JSON array to the path and returns an empty list.

    @param path JSON file path.
    @return Parsed JSON data or an empty list.
    """
    if not os.path.exists(path):
        print(f"[WARN] File does not exist; creating an empty JSON array: {path}")
        write_json(path, [])
        return []

    if os.path.getsize(path) == 0:
        print(f"[WARN] File is empty; initializing it as []: {path}")
        write_json(path, [])
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] File is not valid JSON: {path}\n       -> {e}\n       Initializing it as []")
        write_json(path, [])
        return []


def write_json(path: str, data) -> None:
    """@brief Write JSON with stable UTF-8 formatting.

    @param path Output JSON file path.
    @param data JSON-serializable data.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def pick_original(d: dict):
    """@brief Return the original, non-attacked prompt.

    @param d Dataset item.
    @return Original prompt string or None.
    """
    return (
        d.get("original_prompt")
    )


def pick_prompt(d: dict):
    """@brief Return the adversarial prompt sent to the victim model.

    @param d Dataset item.
    @return Prompt string or None.
    """
    return (
        d.get("prompt")
    )


def process_file(in_file: str, out_dir: str, victim_llm_path: str, use_ollama: bool, ollama_model: str):
    """@brief Run single-prompt inference for one prepared attack file.

    @param in_file Input JSON file containing attacked prompts.
    @param out_dir Output directory.
    @param victim_llm_path Local model path used by vLLM.
    @param use_ollama If True, use Ollama; otherwise use vLLM.
    @param ollama_model Ollama model name or display target model name.
    """
    data = read_json(in_file)
    if not isinstance(data, list):
        print(f"[WARN] {in_file} is not a JSON array; skipping")
        return

    # The LLM object is intentionally created once per file. For vLLM this loads
    # the model into GPU memory; recreating it for every row would be very slow.
    llm = LLM(
        model_path=victim_llm_path,
        temperature=0.0,
        max_tokens=512,
        ollama_model=ollama_model,
        use_ollama=use_ollama,
    )

    out = []
    for item in tqdm(data, desc=f"{Path(in_file).name}", leave=False):
        _id = item.get("id", item.get("idx"))
        original = pick_original(item)
        prompt = pick_prompt(item)

        if not prompt:
            resp = "<<NO PROMPT FOUND>>"
        else:
            try:
                resp = llm.response(prompt)
                time.sleep(0.1)
            except Exception as e:
                resp = f"<<LLM ERROR: {e}>>"

        out.append(
            {
                "id": _id,
                "original_prompt": original,
                "prompt": prompt,
                "response": resp,
            }
        )

    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, os.path.basename(in_file))
    write_json(out_path, out)
    print(f"[OK] {Path(in_file).name} -> {out_path}")


def main():
    """@brief Parse CLI arguments and process one attack JSON file.

    Usage:
      python3 only_attack.py victim_llm_path input_dir output_dir api_ollama_vllm what_ollama_model
    """
    if len(sys.argv) != 6:
        print("Usage: python3 only_attack.py victim_llm_path input_dir output_dir api_ollama_vllm what_ollama_model")
        sys.exit(1)

    victim_llm_path = sys.argv[1]

    dataset_path = sys.argv[2]
    results_dir = sys.argv[3]

    use_ollama      = str2bool(sys.argv[4].lower())
    ollama_model    = sys.argv[5]

    print(f"[INFO] Input file: {dataset_path}")
    print(f"[INFO] Output directory: {results_dir}")
    print(f"[INFO] Model (ollama): {ollama_model}")
    print(f"[INFO] use_ollama:     {use_ollama}")
    process_file(
        in_file=dataset_path,
        out_dir=results_dir,
        victim_llm_path=victim_llm_path,
        use_ollama=use_ollama,
        ollama_model=ollama_model,
    )

    print("\n[DONE] File processed.")


if __name__ == "__main__":
    main()
