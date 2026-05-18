#!/usr/bin/env python3
## @file apply_rules.py
#  @brief Apply a trained rule-tree defense to JSON attack datasets.
#
#  The script loads a defense rule produced by `def.py`, transforms prompts by
#  inserting neutral symbol tokens according to the learned tree parameters, and
#  writes a defended copy of the input dataset. It is intended to be called either
#  directly or through `run_orchestrator.py --defense-apply-rules`.
#
#  @author Bc. Petr Kaska
#  @date 1.2.2026
#
#  Ownership / Contribution statement:
#   - This file was designed and implemented by Bc. Petr Kaska.
#   - The rule loading, anchor interpretation, symbol insertion, deterministic
#     seeding, and dataset rewriting logic are original work by the author.
#   - The implementation uses only standard Python data-processing patterns.

import argparse
import json
import os
import random
from pathlib import Path

# vocab pro "symbols_only" – můžeš si změnit
SYMBOL_VOCAB = list("!”#$%&’()*+,-./:;<=>?@[\]ˆ_‘|~")


def load_rule_tree_from_defense(path: str) -> dict:
    """
    Načte defense.json a vrátí vnitřní objekt rule_tree.
    Pokud soubor rule_tree nemá (nebo chceš rovnou čistý rule_tree),
    zkusí vrátit celý obsah.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # typický případ: defense.json podle tvého příkladu
    if isinstance(data, dict) and "rule_tree" in data:
        return data["rule_tree"]

    # fallback – kdybys někdy předal čistý rule_tree.json
    return data


def generate_symbol_token(k_min: int, k_max: int) -> str:
    length = random.randint(k_min, k_max)
    return "".join(random.choice(SYMBOL_VOCAB) for _ in range(length))


def choose_anchor_indices(n_tokens: int, anchors_cfg: dict, dispersion_cfg: dict):
    quantiles = anchors_cfg.get("quantiles", [])
    delta = anchors_cfg.get("delta", 0.0)
    d_min = dispersion_cfg.get("d_min", 1)

    raw_indices = []
    for q in quantiles:
        jitter = random.uniform(-delta, delta)
        q_j = min(max(q + jitter, 0.0), 1.0)
        idx = int(round(q_j * (n_tokens - 1))) if n_tokens > 0 else 0
        raw_indices.append(idx)

    raw_indices.sort()
    anchor_indices = []
    last_idx = None
    for idx in raw_indices:
        if last_idx is None or abs(idx - last_idx) >= d_min:
            anchor_indices.append(idx)
            last_idx = idx

    return anchor_indices


def mutate_prompt(prompt: str, rule_tree: dict) -> str:
    """Aproximace tvého TreeRule – wrap symboly okolo anchor slov."""
    if not prompt:
        return prompt

    tokens = prompt.split()
    n_tokens = len(tokens)

    anchors_cfg = rule_tree.get("anchors", {})
    dispersion_cfg = rule_tree.get("dispersion", {})
    insert_cfg = rule_tree.get("insertion", {})

    anchor_indices = choose_anchor_indices(n_tokens, anchors_cfg, dispersion_cfg)
    if not anchor_indices:
        return prompt

    budget_rho = insert_cfg.get("budget_rho", 0.0)
    k_min = insert_cfg.get("k_min", 1)
    k_max = insert_cfg.get("k_max", 2)

    max_added_chars = int(len(prompt) * budget_rho)
    added_chars = 0

    new_tokens = []
    anchor_set = set(anchor_indices)

    for idx, tok in enumerate(tokens):
        if idx in anchor_set and added_chars < max_added_chars:
            left = generate_symbol_token(k_min, k_max)
            right = generate_symbol_token(k_min, k_max)
            # mode = "wrap" → symboly před a za původní token
            new_tokens.append(left)
            new_tokens.append(tok)
            new_tokens.append(right)
            added_chars += len(left) + len(right)
        else:
            new_tokens.append(tok)

    return " ".join(new_tokens)


def process_dataset(
    input_dir: str,
    output_dir: str,
    defense_path: str,
    seed: int = 42,
):
    random.seed(seed)

    rule_tree = load_rule_tree_from_defense(defense_path)

    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for json_file in in_path.glob("*.json"):
        with json_file.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Chyba při čtení {json_file}: {e}")
                continue

        # očekávám list objektů s klíčem "prompt" (tvůj formát)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "prompt" in item:
                    item["prompt"] = mutate_prompt(item["prompt"], rule_tree)
        else:
            print(f"Upozornění: {json_file} není list objektů, přeskočeno.")

        out_file = out_path / json_file.name
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"Zpracováno: {json_file.name} -> {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a trained defense rule tree to every JSON dataset in a directory.")
    parser.add_argument(
        "--input-dir",
        default=os.getenv("DEFENSE_APPLY_INPUT_DIR", "/storage/brno2/home/xkaska01/master/prompt_attacker/dataset/my_attacking_dataset"),
        help="Directory with input JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("DEFENSE_APPLY_OUTPUT_DIR", "/storage/brno2/home/xkaska01/master/prompt_attacker/dataset/my_attacking_dataset_with_defense_5"),
        help="Directory where defended JSON files will be written.",
    )
    parser.add_argument(
        "--defense-rule",
        "--rule",
        dest="defense_rule",
        default=os.getenv("DEFENSE_APPLY_RULE", "/storage/brno2/home/xkaska01/master/prompt_attacker/defense/defense_rule_5.json"),
        help="Path to defense_rule JSON or a raw rule_tree JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("DEFENSE_APPLY_SEED", "42")),
        help="Random seed for stochastic token insertion.",
    )
    args = parser.parse_args()

    process_dataset(args.input_dir, args.output_dir, args.defense_rule, seed=args.seed)
