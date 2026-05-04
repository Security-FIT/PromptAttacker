"""Apply a selected defense wrapper to one prepared attack JSON file.

The script is used by `run_orchestrator.py --defense <name>`. It reads a JSON
array of attacked prompts, wraps the victim model with the requested defense,
runs batch inference, and writes the defended responses to the output directory.

Supported defense values:

    rallm, llamaguard, safeguard, ea

The lightweight `ea` defense is a local prompt rewrite and therefore does not
need an additional LLM call. The other defenses wrap an LLM backend through the
shared `attacks.common.llm.LLM` interface.
"""

import argparse
import json
import os
from defense.baseline_defense_llama_guard import LlamaGuard
from defense.baseline_defense_rallm import RALLM
from defense.baseline_defense_safeguard import GoalPrioritizationLLM 

# DefenseEA is optional in some experimental checkouts. Import lazily enough
# that the other defenses can still run even if this file is unavailable.
try:
    from defense.defense_EA import DefenseEA
except Exception:
    DefenseEA = None


def run():
    """@brief Parse CLI arguments, run the selected defense, and write JSON output."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--defense", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--use-ollama")
    parser.add_argument("--per-victim")
    args = parser.parse_args()

    # Select a defense implementation. All branches expose `response_batch()` so
    # the rest of the file can stay independent of the specific defense type.
    if args.defense == "rallm":
        victim = RALLM(use_ollama=args.use_ollama, ollama_model=args.model, per_victim_llm=args.per_victim)
    elif args.defense == "llamaguard":
        victim = LlamaGuard(use_ollama=args.use_ollama, ollama_model=args.model, per_victim_llm=args.per_victim)
    elif args.defense == "safeguard":
        victim = GoalPrioritizationLLM(use_ollama=args.use_ollama, ollama_model=args.model, per_victim_llm=args.per_victim)
    elif args.defense == "ea":
        # DefenseEA is a lightweight prompt-rewrite; adapt it to response_batch.
        if DefenseEA is None:
            raise RuntimeError("DefenseEA not available in defense/defense_EA.py")
        _ea = DefenseEA()

        class _EAWrapper:
            """@brief Adapter exposing DefenseEA through a batch inference API."""

            def __init__(self, ea):
                """@brief Store the wrapped DefenseEA instance.

                @param ea Initialized DefenseEA object.
                """
                self.ea = ea

            def response_batch(self, prompts):
                """@brief Apply the EA prompt rewrite to a batch of prompts.

                @param prompts Input prompts.
                @return Rewritten prompts.
                """
                return [self.ea.apply(p) for p in prompts]

        victim = _EAWrapper(_ea)
    else:
        raise ValueError("Unknown defense")

    # Load prepared attack prompts. The orchestrator passes exactly one JSON
    # file, not a directory.
    with open(args.input, "r") as f:
        data = json.load(f)

    prompts = [item.get("prompt") for item in data]
    
    # Run the defense in batch mode. For LLM-backed defenses this is where the
    # victim model is queried.
    responses = victim.response_batch(prompts)

    # Preserve the evaluation schema and add a `defense` marker for downstream
    # analysis.
    output = []
    for item, resp in zip(data, responses):
        output.append({
            "id": item.get("id"),
            "original_prompt": item.get("original_prompt"),
            "prompt": item.get("prompt"),
            "response": resp,
            "defense": args.defense
        })

    out_path = os.path.join(args.out_dir, os.path.basename(args.input))
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    run()
