# attacks/_4_Rewrite/main.py
"""
Spuštění Rewrite útoku na lokální LLM.

$ python -m attacks._4_Rewrite.main
"""
import os
import json
import pandas
from tqdm import tqdm

from attacks.common.llm import LLM
from attacks._15_Rewrite.attack_rewrite import RewriteAttack
from defense.defense_EA import DefenseEA
from attacks.helpers import load_config                 # už existuje ve tvém repu

def run_rewrite_attack(run_defense: bool = False) -> None:
    """Spustí útok a výsledky uloží do JSONL."""
    defense = DefenseEA()

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configRewrite.yaml")
    cfg         = load_config(config_path)["Rewrite"]

    # — YAML nastavení —
    victim_llm      = cfg["victim_llm"]
    data_path       = cfg["data_path"]
    out_dir         = cfg["output_dict"]
    temperature     = cfg.get("temperature", 0.0)
    max_token       = cfg.get("max_token", 512)
    rewrite_tmpl    = cfg["rewrite_template"]
    begin           = cfg.get("begin", 0)
    end             = cfg.get("end", 100)

    print(f"[INFO] Rewrite: model={victim_llm}, rows=[{begin},{end})")

    # — Data + model —
    adv_bench = pandas.read_csv(data_path)
    llm = LLM(model_path=victim_llm,
              temperature=temperature,
              max_tokens=max_token)

    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, '_15_rewrite.json')

    with open(output_file, "w", encoding="utf-8") as fo:
        for idx, harm_prompt in tqdm(
            enumerate(adv_bench["Goal"][begin:end]), total=end - begin
        ):
            attack_model = RewriteAttack(rewrite_template=rewrite_tmpl)

            log, attack_msgs = attack_model.generate(harm_prompt)

            if run_defense:
                attack_msgs[-1]["content"] = defense(attack_msgs[-1]["content"])

            llm_response = llm.response(attack_msgs)

            fo.write(
                json.dumps(
                    {
                        "id": idx,
                        "original_prompt": log,
                        "prompt": attack_msgs[-1]["content"],
                        "response": llm_response,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fo.flush()

    print(f"[INFO] Results saved to {output_file}")


if __name__ == "__main__":
    run_rewrite_attack(run_defense=False)
