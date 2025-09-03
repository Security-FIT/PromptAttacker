# TENTO FILE JE MUUUUUUUUUUJ

import os
import json
import pandas
from attacks.common.llm import LLM
from tqdm import tqdm
from attacks._1_Cypher.cypher_attack import CypherAttack
from defense.defense_EA import DefenseEA
from attacks.helpers import load_config


def run_cypher_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):
    defense = DefenseEA()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configCypher.yaml")
    cfg = load_config(config_path)
    cfgCypher = cfg["Cypher"]

    # victim_llm = cfgCypher['victim_llm']
    # data_path  = cfgCypher['data_path']
    # out_dir    = cfgCypher['output_dict']
    temperature= cfgCypher.get('temperature', 0.0)
    max_token  = cfgCypher.get('max_token', 512)
    cypher_mode  = cfgCypher.get('cypher_mode', 'WSWR')
    cot        = cfgCypher.get('cot', False)
    lang_gpt   = cfgCypher.get('lang_gpt', False)
    few_shot   = cfgCypher.get('few_shot', False)
    begin      = cfgCypher.get('begin', 0)
    end        = cfgCypher.get('end', 13282)
    # print(f"[INFO] Starting CypherAttack: victim_llm={victim_llm}, flip_mode={cypher_mode}, range=[{begin},{end})")
    # data path


    if dataset_path:
        data_file = dataset_path
        print(f"[INFO] Using data file: {data_file}")

    print(f"[INFO] Loading data from {dataset_path}")
    # init victim llm
    victim_llm = LLM(model_path=victim_llm_path,
               temperature=temperature,
               max_tokens=max_token,
               ollama_model=what_ollama_model,
               use_ollama=api_ollama_vllm, 
               )
    print("[INFO] Initialized LLM client")
    # load data
    adv_bench = pandas.read_csv(dataset_path)

 

    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, '_1_cypher.json')
    entries = []
    with open(output_file, 'w', encoding='utf-8') as fo:

        for id, harm_prompt in tqdm(enumerate(adv_bench["goal"][begin:end])):
            print(f"[INFO] Processing id {id}: {harm_prompt[:50]}...")
            # FlipAttack
            attack_model = CypherAttack(cypher_mode=cypher_mode, 
                                    cot=cot, 
                                    lang_gpt=lang_gpt, 
                                    few_shot=few_shot,
                                    victim_llm=victim_llm)
            
            # generate attack
            log, cypher_attack = attack_model.generate(harm_prompt)
            
            llm_response = victim_llm.response(cypher_attack)
            
            entry = {
                'id': id,
                'cypher_type': attack_model.cypher_mode,
                'original_prompt': log,
                'prompt': cypher_attack[-1]['content'],
                'response': llm_response
            }
            entries.append(entry)

        fo.write(json.dumps(entries, ensure_ascii=False) + '\n')
        fo.flush()



    print(f"[INFO] Results saved to {output_file}")