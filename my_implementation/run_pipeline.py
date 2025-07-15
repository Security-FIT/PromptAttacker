# run.py

import os
import json
import argparse
import yaml

from attacks._1_Cypher.main import run_cypher_attack
from attacks._2_Flip.main import run_flip_attack  
from attacks._3_PiF.PiF_CLM import run_pif_attack
from attacks._4_SQL_StructTransform.main import run_sql_attack
from attacks._5_suffix.main import run_suffix_attack
from attacks._6_Sequential.main import run_sequential_attack
from attacks._7_CitationBreak.main import run_cite_attack
from attacks._8_Bijection.main import run_bijection_attack
from attacks._9_Dialog_completition.main import run_dialog_attack
from attacks._10_Random_Search.main import run_random_attack
from attacks._11_Pair.main import run_pair_attack
from attacks._12_Tap.main import run_tap_attack
from attacks._13_GPT4cypher.main import run_GPTcypher_attack
from attacks._14_Scav.main import run_scav_attack
from attacks._15_Rewrite.main import run_rewrite_attack
from attacks._16_Ica.main import run_ica_attack
from attacks._17_Overload.main import run_overload_attack
from attacks._18_Gcg.main import run_gcg_attack
from attacks._19_Deepinception.main import run_inception_attack
from attacks._20_Base.main import run_base_attack
from attacks._21_Art_Prompt.main import run_artprompt_attack
from attacks._22_Renellm.main import run_renellm_attack
from attacks._25_past.main import run_past_tense_attack
from attacks._24_Autodan.attack_dan import run_autodan_attack

from defense.defense_EA import DefenseEA


def load_config(path):
    ext = os.path.splitext(path)[1].lower()
    with open(path, 'r', encoding='utf-8') as f:
        if ext in ('.yaml', '.yml'):
            return yaml.safe_load(f)
        elif ext == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file: {path}")

if __name__ == "__main__":

    defense = DefenseEA()  
    parser = argparse.ArgumentParser("FlipAttack runner")
    parser.add_argument(
        "--config_file", "-c", type=str, required=True,
        help="Path to YAML (.yaml/.yml) or JSON (.json) config file"
    )
    args = parser.parse_args()

    cfg = load_config(args.config_file)
    
    print(f"[INFO] Running FlipAttack with config: {cfg.get('run_defense', False)}")
    # run_flip_attack(cfg.get('run_defense', False))
    # run_cypher_attack(cfg.get('run_defense', False))
    # run_sql_attack(cfg.get('run_defense', False))
    # run_bijection_attack(cfg.get('run_defense', False))
    # run_suffix_attack(cfg.get('run_defense', False))
    # run_sequential_attack(cfg.get('run_defense', False))
    # run_dialog_attack(cfg.get('run_defense', False))
    # run_cite_attack(cfg.get('run_defense', False))
    # run_ica_attack(cfg.get('run_defense', False))
    # run_renellm_attack(cfg.get('run_defense', False))
    # run_base_attack(cfg.get('run_defense', False))
    # run_artprompt_attack(cfg.get('run_defense', False)) #zatim nefunguje 
    # run_random_attack(cfg.get('run_defense', False))
    # run_pair_attack(cfg.get('run_defense', False))  
    # run_gcg_attack(cfg.get('run_defense', False)) #zatim nefunguje 
    # run_past_tense_attack(cfg.get('run_defense', False))
    # run_tap_attack(cfg.get('run_defense',False))  # Moc slozity asi to nebudu nakonec delat, zatim nefunguje a stravil jsem na nem uz skoro 3 dny
    # run_overload_attack(cfg.get('run_defense', False))
    # run_scav_attack(cfg.get('run_defense', False))
    # run_GPTcypher_attack(cfg.get('run_defense', False))
    # run_autodan_attack(cfg.get('run_defense', False)) 
    # run_inception_attack(cfg.get('run_defense', False))
    # run_rewrite_attack(cfg.get('run_defense', False))  
    print(defense.apply(" some prompt  "))  

    
    # run_pif_attack(cfg.get('run_defense', False))  
    
    # 22 utoku zatim
# python3 run_pipeline.py --config_file config.yaml