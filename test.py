#!/usr/bin/env python3
import os, sys, argparse
from vllm import LLM, SamplingParams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, required=True)
    parser.add_argument("--prompt", "-p", type=str, required=True)
    args = parser.parse_args()

    # ===== diagnostika =====
    abs_path = os.path.abspath(args.model_path)
    print(f"[DEBUG] Zadána cesta:       {args.model_path}")
    print(f"[DEBUG] Absolutní cesta:    {abs_path}")
    if not os.path.exists(abs_path):
        print(f"[ERROR] TAHLE CESTA NEEXISTUJE!", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(abs_path):
        print(f"[ERROR] TAHLE CESTA NENÍ SLOŽKA!", file=sys.stderr)
        sys.exit(1)

    print(f"[DEBUG] Obsah složky “{abs_path}”:")
    for fn in sorted(os.listdir(abs_path)):
        print("   ", fn)
    print("========================\n")

    # ===== vlastní inference =====
    llm = LLM(model=abs_path)
    out = llm.generate(
        [{"prompt": args.prompt}],
        sampling_params=SamplingParams(max_tokens=50, temperature=0.8)
    )
    print("\n--- Odpověď ---")
    # for output in out:
    prompt = out[0].prompt
    generated_text = out[0].outputs[0].text
    print(f"Prompt:    {prompt!r}")
    print(f"Output:    {generated_text!r}")
    print("-" * 60)

if __name__ == "__main__":
    main()

# python3 test.py --model_path models/Llama-2-13b-chat --prompt "Ahoj"