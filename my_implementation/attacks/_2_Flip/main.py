# TENTO FILE JE MUUUUUUUUUUJ

import os
import json
import pandas
from attacks._2_Flip.llm import LLM
from tqdm import tqdm
from attacks._2_Flip.flip_attack import FlipAttack
from defense.defense_EA import DefenseEA
from attacks.helpers import load_config

def run_flip_attack(run_defense: bool = False):

    defense = DefenseEA()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configFlip.yaml")
    cfg = load_config(config_path)
    cfgFlip = cfg["Flip"]
    print(cfg["Flip"])

    victim_llm = cfgFlip['victim_llm']
    data_path  = cfgFlip['data_path']
    out_dir    = cfgFlip['output_dict']
    temperature= cfgFlip.get('temperature', 0.0)
    max_token  = cfgFlip.get('max_token', 512)
    flip_mode  = cfgFlip.get('flip_mode', 'FWO')
    cot        = cfgFlip.get('cot', False)
    lang_gpt   = cfgFlip.get('lang_gpt', False)
    few_shot   = cfgFlip.get('few_shot', False)
    begin      = cfgFlip.get('begin', 0)
    end        = cfgFlip.get('end', 519)
    print(f"[INFO] Starting FlipAttack: victim_llm={victim_llm}, flip_mode={flip_mode}, range=[{begin},{end})")
    # data path


    if data_path:
        data_file = data_path
        print(f"[INFO] Using data file: {data_file}")

    print(f"[INFO] Loading data from {data_path}")
    # init victim llm
    victim_llm = LLM(model_path=victim_llm,
               temperature=temperature,
               max_tokens=max_token)
    print("[INFO] Initialized LLM client")
    # load data
    adv_bench = pandas.read_csv(data_path)

 

    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, '_2_flip.json')

    with open(output_file, 'w', encoding='utf-8') as fo:

        for id, harm_prompt in tqdm(enumerate(adv_bench["goal"][begin:end])):
            print(f"[INFO] Processing id {id}: {harm_prompt[:50]}...")
            # FlipAttack
            attack_model = FlipAttack(flip_mode=flip_mode, 
                                    cot=cot, 
                                    lang_gpt=lang_gpt, 
                                    few_shot=few_shot)
            
            # generate attack
            log, flip_attack = attack_model.generate(harm_prompt)
            
            # attack llms
            if run_defense:
                flip_attack[-1]['content'] = defense(flip_attack[-1]['content'])

            llm_response = victim_llm.response(flip_attack)
            
            entry = {
                'id': id,
                'flip_type': attack_model.flip_mode,
                'original_prompt': log,
                'prompt': flip_attack[-1]['content'],
                'response': llm_response
            }

            fo.write(json.dumps(entry, ensure_ascii=False) + '\n')
            fo.flush()

    print(f"[INFO] Results saved to {output_file}")