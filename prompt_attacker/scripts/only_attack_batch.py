#!/usr/bin/env python3
"""Batch inference variant of only_attack.py.

This script has the same command-line signature as `only_attack.py`, but sends
multiple prompts to the LLM wrapper at once through `response_batch()` when the
backend supports it. It is the preferred runner for large prepared attack
datasets because vLLM can process a batch more efficiently than many separate
single-prompt calls.

Input and output schema:

    input:  [{id, original_prompt, prompt, ...}, ...]
    output: [{id, original_prompt, prompt, response}, ...]
"""

import sys
import os
import json
from tqdm import tqdm
from pathlib import Path
import time

from attacks.common.llm import LLM
from attacks.common.helpers import str2bool


BATCH_SIZE = 4


def ensure_dir(d: str) -> None:
    """@brief Create the output directory if it does not exist.

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
    """@brief Return the clean source prompt stored in an attack item.

    @param d Dataset item.
    @return Original prompt string or None.
    """
    return d.get("original_prompt")


def pick_prompt(d: dict):
    """@brief Return the adversarial prompt sent to the target model.

    @param d Dataset item.
    @return Prompt string or None.
    """
    return d.get("prompt")


def batched_llm_calls(llm: LLM, prompts: list[str]) -> list[str]:
    """@brief Call the LLM in batch mode when available.

    If the shared LLM wrapper exposes `response_batch()`, it is used. Otherwise
    the function falls back to sequential `response()` calls.

    @param llm Initialized LLM wrapper.
    @param prompts Prompts to send to the model.
    @return One response string per prompt.
    """
    if not prompts:
        return []

    # Prefer native batch support when the backend provides it.
    if hasattr(llm, "response_batch") and callable(getattr(llm, "response_batch")):
        try:
            return llm.response_batch(prompts)
        except Exception as e:
            print(f"[WARN] Batch inference failed; falling back to sequential mode: {e}")

    # Sequential fallback is slower but keeps the output schema complete.
    out = []
    for p in prompts:
        try:
            out.append(llm.response(p))
        except Exception as e:
            out.append(f"<<LLM ERROR (single): {e}>>")
    return out


def process_file(in_file: str, out_dir: str, victim_llm_path: str, use_ollama: bool, ollama_model: str):
    """@brief Run batched inference for one prepared attack file.

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

    # vLLM loads the model at construction time, so initialize it only once per
    # input file and then reuse it for all batches.
    llm = LLM(
        model_path=victim_llm_path,
        temperature=0.8,
        max_tokens=512,
        ollama_model=ollama_model,
        use_ollama=use_ollama,
    )

    # Keep the original item order. Only rows with a non-empty prompt are sent
    # to the LLM, and responses are mapped back to their original indices.
    indices = []
    prompts = []
    meta = []  # (id, original_prompt, prompt)

    for idx, item in enumerate(data):
        _id = item.get("id", item.get("idx"))
        original = pick_original(item)
        prompt = pick_prompt(item)

        meta.append((_id, original, prompt))

        if not prompt:
            # Missing prompts are handled when the output is reconstructed.
            continue

        indices.append(idx)
        prompts.append(prompt)

    print(f"[BATCH] Total records: {len(data)}, records with prompt: {len(prompts)}")
    all_responses = [""] * len(data)

    # Process prompts in fixed-size batches.
    for start in tqdm(range(0, len(prompts), BATCH_SIZE), desc=f"{Path(in_file).name} [batch]", leave=False):
        end = min(start + BATCH_SIZE, len(prompts))
        batch_prompts = prompts[start:end]
        batch_indices = indices[start:end]

        batch_resps = batched_llm_calls(llm, batch_prompts)

        # Map responses back to their original item indices.
        for idx_in_data, resp in zip(batch_indices, batch_resps):
            all_responses[idx_in_data] = resp

        # Small pause to avoid overwhelming slower backends.
        time.sleep(0.01)

    # Reconstruct the normalized output schema.
    out = []
    for i, (item_meta) in enumerate(meta):
        _id, original, prompt = item_meta

        if not prompt:
            resp = "<<NO PROMPT FOUND>>"
        else:
            raw_resp = all_responses[i]
            if raw_resp == "":
                resp = "<<NO RESPONSE>>"
            else:
                resp = raw_resp

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
    print(f"[OK-BATCH] {Path(in_file).name} -> {out_path}")


def main():
    """@brief Parse CLI arguments and process one attack JSON file in batches.

    Usage:
      python3 only_attack_batch.py victim_llm_path input_json output_dir api_ollama_vllm what_ollama_model

    The signature intentionally matches `only_attack.py`.
    """
    if len(sys.argv) != 6:
        print("Usage: python3 only_attack_batch.py victim_llm_path input_json output_dir api_ollama_vllm what_ollama_model")
        sys.exit(1)

    victim_llm_path = sys.argv[1]
    dataset_path = sys.argv[2]
    results_dir = sys.argv[3]

    use_ollama = str2bool(sys.argv[4].lower())
    ollama_model = sys.argv[5]

    print(f"[INFO-BATCH] Input file: {dataset_path}")
    print(f"[INFO-BATCH] Output directory: {results_dir}")
    print(f"[INFO-BATCH] Model (ollama/vLLM): {ollama_model}")
    print(f"[INFO-BATCH] use_ollama:         {use_ollama}")
    print(f"[INFO-BATCH] BATCH_SIZE:         {BATCH_SIZE}")

    process_file(
        in_file=dataset_path,
        out_dir=results_dir,
        victim_llm_path=victim_llm_path,
        use_ollama=use_ollama,
        ollama_model=ollama_model,
    )

    print("\n[DONE-BATCH] File processed.")


if __name__ == "__main__":
    main()
