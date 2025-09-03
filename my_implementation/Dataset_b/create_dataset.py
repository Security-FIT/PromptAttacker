#!/usr/bin/env python3
import csv
import json
import os
import sys
from typing import Iterable, Dict, Any

def strip_wrapping_quotes(s: str) -> str:
    if not isinstance(s, str) or len(s) < 2:
        return s
    if (s[0] == s[-1]) and s[0] in {'"', "'"}:
        return s[1:-1]
    return s

def escape_newlines(text: str) -> str:
    if not isinstance(text, str):
        return text
    return text.replace("\n", "\\n")

def iter_jsonl_objects(fp) -> Iterable[Dict[str, Any]]:
    for _, line in enumerate(fp, start=1):
        line = line.strip()
        if not line:
            continue
        if not (line.startswith('{') and line.endswith('}')):
            start = line.find('{')
            end = line.rfind('}')
            if start != -1 and end != -1 and end > start:
                line = line[start:end+1]
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj
        except json.JSONDecodeError:
            continue

def build_csv(root_dir: str, out_csv: str) -> None:
    rows = []
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        raise SystemExit(f"Input directory not found: {root_dir}")

    # Per-model incremental IDs starting at 0
    per_model_id: Dict[str, int] = {}

    subdirs = next(os.walk(root_dir))[1]
    for model_name in sorted(subdirs):
        per_model_id[model_name] = 0
        model_dir = os.path.join(root_dir, model_name)
        for fname in sorted(os.listdir(model_dir)):
            if not fname.lower().endswith('.json'):
                continue
            fpath = os.path.join(model_dir, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    for obj in iter_jsonl_objects(f):
                        original_prompt = strip_wrapping_quotes(obj.get('original_prompt', ''))
                        target_model_answer = obj.get('response', '')
                        original_prompt = escape_newlines(original_prompt)
                        target_model_answer = escape_newlines(target_model_answer)
                        rows.append({
                            'id': per_model_id[model_name],
                            'model': model_name,
                            'original_prompt': original_prompt,
                            'target_model_answer': target_model_answer,
                            'judge_model_score': 'x',
                            'human_score': 'x',
                        })
                        per_model_id[model_name] += 1
            except Exception as e:
                print(f"[WARN] Failed to process {fpath}: {e}", file=sys.stderr)

    fieldnames = ['id', 'model', 'original_prompt', 'target_model_answer', 'judge_model_score', 'human_score']
    with open(out_csv, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")

def main():
    print("Usage: python build_csv_from_dataset.py <input_root_dir> <output_csv>", file=sys.stderr)
    build_csv(
        "/storage/brno2/home/xkaska01/master/my_implementation/Dataset_b",
        "/storage/brno2/home/xkaska01/master/my_implementation/Dataset_b/lol.csv"
    )

if __name__ == '__main__':
    main()
