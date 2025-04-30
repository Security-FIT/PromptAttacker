#!/usr/bin/env python3
"""
Skeleton for a universal pipeline to load attacks, defenses and evaluators based on a YAML config.
Directory structure:

my_implementation/
├── attacks/            # each attack in its own subfolder as a .py module
│   ├── Flip/Flip.py
│   └── PiF_ICLR/PiF.py
├── defense/            # your defense module(s)
│   └── defense_EA.py
├── evaluate/           # your evaluator module(s)
│   └── evaluator.py
├── models/             # optional: model wrappers or clients
├── dataset/            # input data
│   └── data.txt
├── config.yaml         # pipeline configuration
└── run_pipeline.py     # this script

"""
import os
import yaml
import argparse
import importlib.util


def load_module_from_path(name, path):
    """
    Dynamically load a Python module from a given file path.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(config_path: str):
    # 1. Read configuration
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    input_path  = cfg['input']
    output_dir  = cfg['output']
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load input data
    with open(input_path, 'r') as f:
        data = [line.strip() for line in f if line.strip()]

    # 3. Run attacks
    attack_results = {}
    for attack_name, attack_cfg in cfg.get('attacks', {}).items():
        path   = attack_cfg['path']
        params = attack_cfg.get('params', {})
        module = load_module_from_path(attack_name, path)

        # assume each attack module exports `attack(text: str, **kwargs) -> str`
        attacked = [module.attack(sample, **params) for sample in data]
        attack_results[attack_name] = attacked

    # 4. (Optional) Run defense
    defended_results = {}
    if cfg.get('run_defense', False):
        def_cfg   = cfg['defense']
        def_mod   = load_module_from_path('defense', def_cfg['path'])
        def_params = def_cfg.get('params', {})

        # assume `defense.defend(text: str, **kwargs) -> str`
        for name, attacked in attack_results.items():
            defended = [def_mod.defend(a, **def_params) for a in attacked]
            defended_results[f"{name}_defended"] = defended

    # 5. (Optional) Run evaluator
    eval_results = {}
    if 'evaluator' in cfg and cfg['evaluator']:
        eval_cfg = cfg['evaluator']
        ev_mod   = load_module_from_path('evaluator', eval_cfg['path'])
        ev_params = eval_cfg.get('params', {})

        # assume `evaluator.evaluate(list_of_texts: List[str], **kwargs) -> Any`
        target_sets = {**attack_results, **defended_results}
        for name, texts in target_sets.items():
            eval_results[name] = ev_mod.evaluate(texts, **ev_params)

    # 6. Save outputs
    def save(name, items):
        out_file = os.path.join(output_dir, f"{name}.txt")
        with open(out_file, 'w') as f:
            for i in items:
                f.write(f"{i}\n")

    for name, res in attack_results.items():
        save(name, res)
    for name, res in defended_results.items():
        save(name, res)

    # if evaluator returns scalar or dict, dump as YAML for readability
    if eval_results:
        with open(os.path.join(output_dir, 'evaluation_results.yaml'), 'w') as f:
            yaml.safe_dump(eval_results, f)

    print(f"Pipeline complete. Results in '{output_dir}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run attack/defense/evaluation pipeline')
    parser.add_argument('--config', '-c', default='config.yaml', help='Path to config YAML')
    args = parser.parse_args()
    main(args.config)