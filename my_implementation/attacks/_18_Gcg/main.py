## @file main.py
#  @brief Runner script for the GCG (Greedy Coordinate Gradient) adversarial suffix attack
#
#  This script runs a GCG-based adversarial suffix (control-string) attack against
#  a local HuggingFace causal language model. It supports optional suffix
#  optimization (training) on a dedicated dataset and then evaluates the learned
#  suffix on a target dataset, saving results as JSON Lines.
#
#  @author Bc. Petr Kaška
#  @date 1.2.2026
#
#  Ownership / Contribution statement:
#   - This runner script was fully implemented by Bc. Petr Kaška.
#   - The orchestration logic, configuration handling, synchronous worker,
#     suffix training pipeline, evaluation loop, and result serialization are
#     original work by the author.
#   - The runner integrates GCG helper classes (AttackPrompt/PromptManager/
#     MultiPromptAttack) from the local framework modules under `attacks/_18_Gcg`.
#
#  Research basis:
#   - The adversarial suffix optimization methodology (GCG / control-string search)
#     is based on:
#       "Universal and Transferable Adversarial Attacks on Aligned Language Models"
#       arXiv:2307.15043v2 (Zou et al., 2023)
#       https://arxiv.org/abs/2307.15043v2
#

import os, json, yaml, pandas as pd
from pathlib import Path

from attacks.common.helpers import load_config, str2bool
import torch, sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.model import get_conversation_template
import time, gc

from attacks._18_Gcg.helpers.gcg_attack import (
    GCGAttackPrompt       as AttackPrompt,
    GCGPromptManager      as PromptManager,
    GCGMultiPromptAttack  as MultiPromptAttack,
)

from attacks._18_Gcg.helpers.attack_manager import get_embedding_layer, get_embeddings

import queue
class _SyncWorker:
    def __init__(self, model, tokenizer, conv_template):
        self.model          = model
        self.tokenizer      = tokenizer
        self.conv_template  = conv_template
        self.results        = queue.Queue()        

    def __call__(self, prompt_obj, mode, model, *args, **kwargs):
        if mode == "grad":
            grad = prompt_obj.grad(model).detach().cpu()
            self.results.put(grad)
        elif mode == "logits":
            logits, ids = prompt_obj.logits(model, *args, **kwargs)
            self.results.put((logits.detach().cpu(),ids.detach().cpu()))
        elif mode == "test":
            self.results.put(prompt_obj.test(model))
        elif mode == "test_loss":
            self.results.put(prompt_obj.test_loss(model))
        else:
            raise ValueError(f"Unknown mode '{mode}'")

def _train_suffix(cfg, tokenizer, model, conv_templ):
    print("[INFO] → Spouštím trénink suffixu")
    df = pd.read_csv(cfg["train_dataset"])
    goals   = df["goal"  ].astype(str).tolist()
    targets = df["target"].astype(str).tolist()

    worker   = _SyncWorker(model, tokenizer, conv_templ)
    managers = {"PM": PromptManager, "AP": AttackPrompt, "MPA": MultiPromptAttack}

    mpa = MultiPromptAttack(
        goals, targets, [worker],
        control_init = cfg.get("control_init", "! ! ! ! ! "),
        managers     = managers
    )

    best_suffix = mpa.run(
        n_steps        = cfg.get("train_steps"),
        batch_size     = cfg.get("train_batch_size"),
        topk           = cfg.get("train_topk",        256),
        temp           = cfg.get("train_temp",        1.0),
        allow_non_ascii= cfg.get("allow_non_ascii",  False),
        verbose        = True
    )

    print(f"[INFO] ✓ Trénink hotový, nejlepší suffix:\n    '{best_suffix}'")
    return best_suffix[0] if isinstance(best_suffix, tuple) else best_suffix

def run_gcg_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(script_dir + "/configGCG.yaml")["GCG"]

    template     = cfg.get("conversation_template", "vicuna")
    device       = cfg.get("device", "cuda:0")
    temperature  = cfg.get("temperature", 0.0)
    max_token    = cfg.get("max_token", 512)
    suffix       = cfg.get("control_init")
    begin        = cfg.get("begin", 0)
    end          = cfg.get("end", None)


    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, '_18_gcg.json')

    print(f"[INFO] GCG‑Attack • model={victim_llm_path} • template={template}")

    tokenizer = AutoTokenizer.from_pretrained(victim_llm_path, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
                    victim_llm_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ).to(device).eval()
    conv_template = get_conversation_template(template)

    if cfg.get("optimize_suffix"):
        t0      = time.time()
        suffix  = _train_suffix(cfg, tokenizer, model, conv_template)
        cfg["control_init"] = suffix       
        print(f"[INFO] Trénink trval {time.time()-t0:.1f}s")
        gc.collect(); torch.cuda.empty_cache()

    df = pd.read_csv(dataset_path)
    if end is None: end = len(df)

    print(f"[INFO] Output file: {out_file}\n")
    with open(out_file, "w", encoding="utf‑8") as fo:
        for idx, row in df.iloc[begin:end].iterrows():
            prompt_obj = AttackPrompt(
                goal   = row.get("goal", ""),          
                target = row["target"],                
                tokenizer      = tokenizer,
                conv_template  = conv_template,
                control_init   = suffix
            )


            prompt_text = prompt_obj.eval_str        

            with torch.no_grad():
                reply = prompt_obj.generate_str(model)

            fo.write(json.dumps({
                "id": int(idx),
                "original_prompt": row.get("goal", ""),
                "prompt": prompt_text,
                "response": reply
            }, ensure_ascii=False) + "\n")
            fo.flush()

    print(f"[INFO] Výstup uložen → {out_file}")



if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 run_cypher.py victim_llm_path results_dir dataset_path api_ollama_vllm what_ollama_model")
        sys.exit(1)

    victim_llm_path = sys.argv[1]
    results_dir = sys.argv[2]
    dataset_path = sys.argv[3]
    api_ollama_vllm = str2bool(sys.argv[4].lower())
    what_ollama_model = sys.argv[5]

    run_gcg_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)