import os
import json
import pandas as pd
from attacks._6_Sequential.llm import LLM
from attacks._6_Sequential.sequential_attack import SequentialAttack   
from attacks.helpers import load_config
from defense.defense_EA import DefenseEA
from tqdm import tqdm


def run_sequential_attack(run_defense: bool = False):

    defense = DefenseEA()
    # Load configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configSequential.yaml")
    cfg = load_config(config_path)
    cfgSeq = cfg.get("Sequential", {})
    print(cfgSeq)

    # Extract settings
    victim_model = cfgSeq.get('victim_llm')
    data_path    = cfgSeq.get('data_path')
    out_dir      = cfgSeq.get('output_dir')
    temperature  = cfgSeq.get('temperature', 0.0)
    max_tokens   = cfgSeq.get('max_tokens', 512)
    steps        = cfgSeq.get('steps', 3)
    begin        = cfgSeq.get('begin', 0)
    end          = cfgSeq.get('end', None)

    # Initialize LLM client
    llm_client = LLM(model_path=victim_model, temperature=temperature, max_tokens=max_tokens)
    print("[INFO] Initialized LLM client")

    # Load data
    print(f"[INFO] Loading data from {data_path}")
    df = pd.read_csv(data_path)
    if end is None:
        end = len(df)

    # Prepare output
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, '_6_sequential_attack.json')

    with open(output_file, 'w', encoding='utf-8') as fo:
        for idx, harm_prompt in tqdm(enumerate(df['goal'][begin:end]), total=end-begin):
            print(f"[INFO] Processing id {idx}: {harm_prompt[:50]}...")
            # Build the sequential attack prompts
            attack_model = SequentialAttack(steps)
            log, sequence_prompts = attack_model.generate(harm_prompt)

            last_prompt = sequence_prompts[-1]['user']
            response = llm_client.response([
                {'role': 'system', 'content': sequence_prompts[-1]['system']},
                {'role': 'user', 'content': last_prompt}
            ])

            entry = {
                'id': idx,
                'original_prompt': log,
                'prompt': last_prompt,
                'response': response
            }
            fo.write(json.dumps(entry, ensure_ascii=False) + '')
            fo.flush()

    print(f"[INFO] Results saved to {output_file}")

