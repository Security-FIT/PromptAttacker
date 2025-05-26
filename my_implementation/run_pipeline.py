# run.py

import os
import json
import argparse
import yaml

from attacks.Flip.main import run_flip_attack  
from attacks.Cypher.main import run_cypher_attack

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
    run_flip_attack(cfg.get('run_defense', False))
    # run_cypher_attack(cfg.get('run_defense', False))


# python3 run_pipeline.py --config_file config.yaml