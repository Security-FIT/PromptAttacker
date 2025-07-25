import os
import json
import pandas as pd
from attacks.common.llm import LLM
from attacks._6_Sequential.sequential_attack import SequentialAttack   
from attacks.helpers import load_config
from defense.defense_EA import DefenseEA
from tqdm import tqdm


def run_sequential_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):

    defense = DefenseEA()
    # Load configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configSequential.yaml")
    cfg = load_config(config_path)
    cfgSeq = cfg.get("Sequential", {})
    print(cfgSeq)

    # Extract settings
    # victim_model = cfgSeq.get('victim_llm')
    # data_path    = cfgSeq.get('data_path')
    # out_dir      = cfgSeq.get('output_dir')
    temperature  = cfgSeq.get('temperature', 0.0)
    max_tokens   = cfgSeq.get('max_tokens', 512)
    steps        = cfgSeq.get('steps', 3)
    begin        = cfgSeq.get('begin', 0)
    end          = cfgSeq.get('end', None)

    # Initialize LLM client
    llm_client = LLM(model_path=victim_llm_path, temperature=temperature, max_tokens=max_tokens,         ollama_model=what_ollama_model,
        use_ollama=api_ollama_vllm)
    print("[INFO] Initialized LLM client")

    # Load data
    print(f"[INFO] Loading data from {dataset_path}")
    df = pd.read_csv(dataset_path)
    if end is None:
        end = len(df)

    # Prepare output
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, '_6_sequential_attack.json')

    with open(output_file, 'w', encoding='utf-8') as fo:
        for idx, harm_prompt in tqdm(enumerate(df['goal'][begin:end]), total=end-begin):
            print(f"[INFO] Processing id {idx}: {harm_prompt[:50]}...")
            

            attack_model = SequentialAttack(steps)
            log, sequence_prompts = attack_model.generate(harm_prompt)

            all_system = " ".join(item['system'] for item in sequence_prompts)


            all_user   = " ".join(item['user']   for item in sequence_prompts)

            response = llm_client.response([
                {'role': 'system', 'content': all_system},
                {'role': 'user', 'content': all_user}
            ])

            entry = {
                'id': idx,
                'original_prompt': log,
                'prompt': all_user,
                'response': response
            }
            fo.write(json.dumps(entry, ensure_ascii=False) + '')
            fo.flush()

    print(f"[INFO] Results saved to {output_file}")

