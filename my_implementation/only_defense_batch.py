import argparse
import json
import os
from attacks.common.llm import LLM
from defense.baseline_defense_llama_guard import LlamaGuardLLM
from defense.baseline_defense_rallm import RALLM
from defense.baseline_defense_safeguard import GoalPrioritizationLLM 

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--defense", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--use-ollama")
    parser.add_argument("--per-victim")
    args = parser.parse_args()

    # 2. Obalení obranou
    if args.defense == "rallm":
        victim = RALLM(use_ollama=args.use_ollama, ollama_model=args.model, per_victim_llm=args.per_victim)
    elif args.defense == "llamaguard":
        victim = LlamaGuardLLM(use_ollama=args.use_ollama, ollama_model=args.model, per_victim_llm=args.per_victim)
    elif args.defense == "safeguard":
        victim = GoalPrioritizationLLM(use_ollama=args.use_ollama, ollama_model=args.model, per_victim_llm=args.per_victim)
    else:
        raise ValueError("Unknown defense")

    # 3. Načtení dat
    with open(args.input, "r") as f:
        data = json.load(f)

    prompts = [item.get("prompt") for item in data]
    
    # 4. Spuštění batch inference (využije tvůj implementovaný response_batch)
    responses = victim.response_batch(prompts)

    # 5. Uložení výsledků
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