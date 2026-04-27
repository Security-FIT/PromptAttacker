#!/usr/bin/env python3
import textwrap

def job_template(name, cmd, ollama_model, homedir="/storage/brno2/home/xkaska01/master/my_implementation"):
    return textwrap.dedent(f"""\
        #!/bin/bash
        #PBS -q default@pbs-m1.metacentrum.cz
        #PBS -N {name}_{ollama_model}
        #PBS -l select=1:ncpus=1:ngpus=1:mem=20gb:gpu_mem=40gb
        #PBS -l walltime=8:00:00

        HOMEDIR={homedir}
        export PYTHONPATH="$(pwd):$PYTHONPATH"
        cd $HOMEDIR

        export CUDA_VISIBLE_DEVICES=0
        module add mambaforge
        mamba activate /storage/brno2/home/xkaska01/.conda/envs/diplomka

        # start ollama (background) and ensure model
        /storage/brno2/home/xkaska01/test/bin/ollama  serve > $HOMEDIR/ollama.log 2>&1 &
        /storage/brno2/home/xkaska01/test/bin/ollama  pull {ollama_model}
        python3 -m pip install --user nltk

        {cmd}

        echo "End {name}: $(date)"
    """)


def batch_template(name, cmd, ollama_model, homedir="/storage/brno2/home/xkaska01/master/my_implementation"):
    return textwrap.dedent(f"""\
        #!/bin/bash
        #PBS -q default@pbs-m1.metacentrum.cz
        #PBS -N {name}_{ollama_model}
        #PBS -l select=1:ncpus=1:ngpus=1:mem=20gb:gpu_mem=50gb
        #PBS -l walltime=12:00:00

        HOMEDIR={homedir}
        export PYTHONPATH="$(pwd):$PYTHONPATH"
        cd $HOMEDIR

        export CUDA_VISIBLE_DEVICES=0
        module add mambaforge
        mamba activate /storage/brno2/home/xkaska01/.conda/envs/diplomka

        python3 -m pip install --user nltk

        {cmd}

        echo "End {name}: $(date)"
    """)


def results_eval_template(name, cmd, homedir="/storage/brno2/home/xkaska01/master/my_implementation"):
    return textwrap.dedent(f"""\
        #!/bin/bash
        #PBS -q default@pbs-m1.metacentrum.cz
        #PBS -N {name}
        #PBS -l select=1:ncpus=1:ngpus=1:mem=50gb:gpu_mem=30gb
        #PBS -l walltime=15:00:00

        HOMEDIR={homedir}
        export PYTHONPATH="$(pwd):$PYTHONPATH"
        cd $HOMEDIR

        export CUDA_VISIBLE_DEVICES=0
        module add mambaforge
        mamba activate /storage/brno2/home/xkaska01/.conda/envs/diplomka

        /storage/brno2/home/xkaska01/test/bin/ollama serve > $HOMEDIR/ollama.log 2>&1 &

        # wait for Ollama
        for i in $(seq 1 60); do
            if curl -s http://127.0.0.1:11434/api/tags > /dev/null; then
                break
            fi
            sleep 2
        done

        {cmd}

        echo "End {name}: $(date)"
    """)
