#!/usr/bin/env python3
import os, yaml, textwrap, subprocess, argparse
# from my_implementation.evaluate.evaluate_full_results_datasets import evaluate_model

# === Nastavení ===
CONFIG = "/storage/brno2/home/xkaska01/master/my_implementation/config.yaml"  # uprav dle sebe
SUBMIT = True

# Moduly (ty, co máš) – spouští se jako: python3 -m <module> <args>
ATTACK_MODULES = [
    "attacks._1_Cypher.main",
    "attacks._2_Flip.main",
    # "attacks._3_PiF.PiF_CLM",
    "attacks._4_SQL_StructTransform.main",
    "attacks._5_suffix.main",
    "attacks._6_Sequential.main",
    "attacks._7_CitationBreak.main",
    "attacks._8_Bijection.main",
    "attacks._9_Dialog_completition.main",
    "attacks._10_Random_Search.main",
    "attacks._11_Pair.main", # tak za 10 h by to mohl dat
    "attacks._12_Tap.main",
    "attacks._13_GPT4cypher.main",
    "attacks._14_MultiLang.main",
    "attacks._15_Rewrite.main",
    "attacks._16_Ica.main",
    "attacks._17_Overload.main",
    "attacks._18_Gcg.main",
    "attacks._19_Deepinception.main",
    "attacks._20_Base.main",
    "attacks._21_Art_Prompt.main",
    "attacks._22_Renellm.main",
    "attacks._24_Autodan.main",
    "attacks._25_past.main",
    "attacks._26_Chameleon.main",
]

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def job_script_content(name, cmd, ollama_model):

    gpu = "45gb"
    cpu = "200gb"
    ngpu = 1
    ncpu = 1
    walltime = "6:00:00"

    # if name in "_12_Tap":
    #     walltime = "18:00:00"
    # elif name in "_11_Pair":
    #     walltime = ":00:00"
    # elif name in "_3_PiF_PiF_CLM":
    #     walltime = "12:00:00"
    # elif name in "_25_past":
    #     walltime = "18:00:00"
    # print(name)
    # print(walltime)
    # exit(0)
    return textwrap.dedent(f"""\
        #!/bin/bash
        #PBS -q default@pbs-m1.metacentrum.cz
        #PBS -N {name}_{ollama_model}
        #PBS -l select=1:ncpus={ncpu}:ngpus={ngpu}:mem={cpu}:gpu_mem={gpu}
        #PBS -l walltime={walltime}

        HOMEDIR=/storage/brno2/home/xkaska01/master/my_implementation

        export PYTHONPATH="$(pwd):$PYTHONPATH"
        cd $HOMEDIR
        
        export CUDA_VISIBLE_DEVICES=0
        module add mambaforge
        mamba activate /storage/brno2/home/xkaska01/.conda/envs/diplomka

        # spustit ollama (upravit cesty, pokud je potřeba)
        # spouštím jako background, loguji výstup
        /storage/brno2/home/xkaska01/test/bin/ollama serve > $HOMEDIR/ollama.log 2>&1 &
        /storage/brno2/home/xkaska01/test/bin/ollama pull qwen2.5:7b
        /storage/brno2/home/xkaska01/test/bin/ollama pull {ollama_model}
        python3 -m pip install --user nltk

        {cmd}

        echo "End {name}: $(date)"
    """)


def results_eval_all(name):

    gpu = "15gb"
    cpu = "100gb"
    ngpu = 1
    ncpu = 1
    walltime = "5:00:00"

    # print(name)
    # print(walltime)
    # exit(0)
    return textwrap.dedent(f"""\
        #!/bin/bash
        #PBS -q default@pbs-m1.metacentrum.cz
        #PBS -N {name}
        #PBS -l select=1:ncpus={ncpu}:ngpus={ngpu}:mem={cpu}:gpu_mem={gpu}
        #PBS -l walltime={walltime}

        HOMEDIR=/storage/brno2/home/xkaska01/master/my_implementation

        export PYTHONPATH="$(pwd):$PYTHONPATH"
        cd $HOMEDIR
        
        export CUDA_VISIBLE_DEVICES=0
        module add mambaforge
        mamba activate /storage/brno2/home/xkaska01/.conda/envs/diplomka

        # spustit ollama (upravit cesty, pokud je potřeba)
        # spouštím jako background, loguji výstup
        /storage/brno2/home/xkaska01/test/bin/ollama serve > $HOMEDIR/ollama.log 2>&1 &
        /storage/brno2/home/xkaska01/test/bin/ollama pull gemma3:12b
        python3 -m pip install --user nltk

        python3 /storage/brno2/home/xkaska01/master/my_implementation/evaluate/evaluate_full_results_datasets.py {name}

        echo "End {name}: $(date)"
    """)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Spustit evaluaci všech výsledků před útoky")
    args = parser.parse_args()


    cfg = load_cfg(CONFIG)
    victim_llm = cfg["victim_llm"]
    results_dir = cfg["results_dir"]
    dataset = cfg["dataset_path"]
    use_ollama = str(cfg.get("use_ollama", True)).lower()  # předáme jako text "true/false"
    ollama_model = cfg["ollama_model"]

    jobs_dir = os.path.join(results_dir, "jobs")
    ensure_dir(jobs_dir)
    ensure_dir(results_dir)

    print(f"[INFO] Results dir: {results_dir}")
    print(f"[INFO] Jobs dir:    {jobs_dir}")

    if args.eval:
        print("[INFO] Spouštím evaluaci všech výsledků před útoky...")
        model = "internlm2.5:latest"
        script_path = os.path.join("/storage/brno2/home/xkaska01/master/my_implementation/evaluate/results_eval", f"job_{model}.sh")
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(results_eval_all(model))
        os.chmod(script_path, 0o755)
        # created.append(script_path)
        print(f"[INFO] Created {script_path}")

        if SUBMIT:
            try:
                res = subprocess.run(["qsub", script_path], check=True, capture_output=True, text=True)
                print(f"  -> qsub: {res.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"  !! qsub error: {e.stderr.strip()}")

        return

    
    created = []
    for module in ATTACK_MODULES:
        name = module.split(".")[1]  # např. "_1_Cypher" → můžeš si to přejmenovat, klidně zjemni:
        safe_name = module.replace("attacks.", "").replace(".main", "").replace(".", "_")
        # příkaz: python3 -m attacks.<...>.main <victim_llm> <results_dir> <dataset> <use_ollama> <ollama_model>
        cmd = f"python3 -m {module} {victim_llm} {results_dir} {dataset} {use_ollama} {ollama_model}"

        script_path = os.path.join(jobs_dir, f"job_{safe_name}.sh")
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(job_script_content(safe_name, cmd, ollama_model))
        os.chmod(script_path, 0o755)
        created.append(script_path)
        print(f"[INFO] Created {script_path}")

        if SUBMIT:
            try:
                res = subprocess.run(["qsub", script_path], check=True, capture_output=True, text=True)
                print(f"  -> qsub: {res.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"  !! qsub error: {e.stderr.strip()}")

    print(f"[INFO] Hotovo. Vytvořeno jobů: {len(created)}")

if __name__ == "__main__":
    main()
