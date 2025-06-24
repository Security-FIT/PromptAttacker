import os
import json
import pandas as pd
from attacks._9_Dialog_completition.llm import LLM
from attacks._9_Dialog_completition.dialog_attack import DialogCompletionAttack   
from attacks.helpers import load_config
from tqdm import tqdm


def run_dialog_attack(run_defense: bool = False):
    # Load configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configDialog.yaml")
    cfg = load_config(config_path)
    cfgSeq = cfg.get("Dialog", {})
    print(cfgSeq)

    # Extract settings
    victim_model = cfgSeq.get('victim_llm')
    data_path    = cfgSeq.get('data_path')
    out_dir      = cfgSeq.get('output_dir')
    temperature  = cfgSeq.get('temperature', 0.0)
    max_tokens   = cfgSeq.get('max_tokens', 512)
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
    output_file = os.path.join(out_dir, '_9_dialog_attack.json')

    with open(output_file, 'w', encoding='utf-8') as fo:
        for idx, harm_prompt in tqdm(enumerate(df['goal'][begin:end]), total=end-begin):
            print(f"[INFO] Processing id {idx}: {harm_prompt[:50]}...")
            # Build the sequential attack prompts
            attack_model = DialogCompletionAttack()
            log, sequence_prompts = attack_model.generate(harm_prompt)

            last_prompt = sequence_prompts[-1]['user']
            all_system = " ".join(item['system'] for item in sequence_prompts)

            # Spojení všech 'user' do jednoho stringu
            all_user   = " ".join(item['user']   for item in sequence_prompts)

            print("Combined SYSTEM:\n", all_system)
            print("\nCombined USER:\n", all_user)
            # exit(0)
            # Execute sequence and keep only final turn
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

