# run.py

import os
import json
import argparse
import yaml

from attacks._1_Cypher.main import run_cypher_attack
from attacks._2_Flip.main import run_flip_attack  
from attacks._4_SQL_StructTransform.main import run_sql_attack
from attacks._5_suffix.main import run_suffix_attack
from attacks._6_Sequential.main import run_sequential_attack
from attacks._7_CitationBreak.main import run_cite_attack
from attacks._8_Bijection.main import run_bijection_attack
from attacks._9_Dialog_completition.main import run_dialog_attack
from attacks._16_Ica.main import run_ica_attack
from attacks._22_Renellm.main import run_renellm_attack

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
    run_renellm_attack(cfg.get('run_defense', False))
# python3 run_pipeline.py --config_file config.yaml