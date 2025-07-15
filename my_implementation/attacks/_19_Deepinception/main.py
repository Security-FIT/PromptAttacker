# attacks/_3_DeepInception/main.py
"""
Spuštění Deep-Inception útoku na lokálním LLM.

$ python -m attacks._3_DeepInception.main
"""
import os
import json
import pandas
from tqdm import tqdm

# Re-use existující wrapper (nemusíš duplikovat kód)
from attacks._19_Deepinception.llm import LLM          # stejné jako dříve :contentReference[oaicite:10]{index=10}
from attacks._19_Deepinception.deep_inception_attack import DeepInceptionAttack
from defense.defense_EA import DefenseEA
from attacks.helpers import load_config      

def run_inception_attack(run_defense: bool = False) -> None:
    """Spustí útok a uloží výsledky do JSONL."""
    defense = DefenseEA()

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configInception.yaml")
    cfg         = load_config(config_path)["DeepInception"]

    # — Nastavení z Yaml —
    victim_llm   = cfg["victim_llm"]
    data_path    = cfg["data_path"]
    out_dir      = cfg["output_dict"]
    temperature  = cfg.get("temperature", 0.0)
    max_token    = cfg.get("max_token", 512)
    scene        = cfg.get("scene", "science fiction")
    char_num     = cfg.get("character_number", 5)
    layer_num    = cfg.get("layer_number", 5)
    begin        = cfg.get("begin", 0)
    end          = cfg.get("end", 100)

    print(f"[INFO] DeepInception: model={victim_llm}, rows=[{begin},{end})")

    # — Načti data a model —
    adv_bench = pandas.read_csv(data_path)
    llm = LLM(model_path=victim_llm,
              temperature=temperature,
              max_tokens=max_token)

    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, '_19_deep_inception.json')

    with open(output_file, "w", encoding="utf-8") as fo:
        for idx, harm_prompt in tqdm(
            enumerate(adv_bench["Goal"][begin:end]), total=end - begin
        ):
            attack_model = DeepInceptionAttack(
                scene=scene,
                character_number=char_num,
                layer_number=layer_num,
            )

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
