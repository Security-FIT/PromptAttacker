#!/usr/bin/env python3
"""PBS job templates used by run_orchestrator.py.

The orchestrator does not execute heavy model inference directly in normal
mode. Instead it renders one of these shell scripts and submits it through
`qsub`. Keeping the PBS parameters in one file makes it easier to tune GPU
memory, walltime, conda activation, Ollama startup, and CUDA compatibility for
the whole project.

The `gpu_cap=cuda80` constraint is intentional. The current environment's
PyTorch build supports CUDA architectures up to sm_90, while newer Blackwell
GPUs expose sm_120 and fail with "no kernel image is available for execution on
the device". Requesting cuda80 avoids those incompatible nodes.
"""

import textwrap


def job_template(name, cmd, ollama_model, homedir="/storage/brno2/home/xkaska01/master/my_implementation"):
    """@brief Render a PBS script for jobs that need a local Ollama server.

    @param name PBS job name prefix.
    @param cmd Shell command executed inside the job.
    @param ollama_model Ollama model name used by the job.
    @param homedir Project directory on the cluster filesystem.
    @return Complete PBS shell script.
    """
    return textwrap.dedent(f"""\
        #!/bin/bash
        #PBS -q default@pbs-m1.metacentrum.cz
        #PBS -N {name}_{ollama_model}
        #PBS -l select=1:ncpus=1:ngpus=1:mem=20gb:gpu_mem=40gb:gpu_cap=cuda80
        #PBS -l walltime=8:00:00

        HOMEDIR={homedir}
        cd $HOMEDIR
        export PYTHONPATH="$HOMEDIR:$PYTHONPATH"

        export CUDA_VISIBLE_DEVICES=0
        module add mambaforge
        mamba activate /storage/brno2/home/xkaska01/.conda/envs/diplomka

        # Start Ollama in the background and ensure the requested model exists.
        /storage/brno2/home/xkaska01/test/bin/ollama  serve > $HOMEDIR/ollama.log 2>&1 &
        /storage/brno2/home/xkaska01/test/bin/ollama  pull {ollama_model}
        python3 -m pip install --user nltk

        {cmd}

        echo "End {name}: $(date)"
    """)


def batch_template(name, cmd, ollama_model, homedir="/storage/brno2/home/xkaska01/master/my_implementation"):
    """@brief Render a PBS script for vLLM/batch jobs.

    This template does not start Ollama. It is used for local vLLM inference and
    simple utility scripts.

    @param name PBS job name prefix.
    @param cmd Shell command executed inside the job.
    @param ollama_model Model name included in the PBS job name.
    @param homedir Project directory on the cluster filesystem.
    @return Complete PBS shell script.
    """
    return textwrap.dedent(f"""\
        #!/bin/bash
        #PBS -q default@pbs-m1.metacentrum.cz
        #PBS -N {name}_{ollama_model}
        #PBS -l select=1:ncpus=1:ngpus=1:mem=20gb:gpu_mem=50gb:gpu_cap=cuda80
        #PBS -l walltime=12:00:00

        HOMEDIR={homedir}
        cd $HOMEDIR
        export PYTHONPATH="$HOMEDIR:$PYTHONPATH"

        export CUDA_VISIBLE_DEVICES=0
        module add mambaforge
        mamba activate /storage/brno2/home/xkaska01/.conda/envs/diplomka

        python3 -m pip install --user nltk

        {cmd}

        echo "End {name}: $(date)"
    """)


def results_eval_template(name, cmd, homedir="/storage/brno2/home/xkaska01/master/my_implementation"):
    """@brief Render a PBS script for evaluation jobs.

    @param name PBS job name.
    @param cmd Shell command executed inside the job.
    @param homedir Project directory on the cluster filesystem.
    @return Complete PBS shell script.
    """
    return textwrap.dedent(f"""\
        #!/bin/bash
        #PBS -q default@pbs-m1.metacentrum.cz
        #PBS -N {name}
        #PBS -l select=1:ncpus=1:ngpus=1:mem=50gb:gpu_mem=30gb:gpu_cap=cuda80
        #PBS -l walltime=15:00:00

        HOMEDIR={homedir}
        cd $HOMEDIR
        export PYTHONPATH="$HOMEDIR:$PYTHONPATH"

        export CUDA_VISIBLE_DEVICES=0
        module add mambaforge
        mamba activate /storage/brno2/home/xkaska01/.conda/envs/diplomka

        /storage/brno2/home/xkaska01/test/bin/ollama serve > $HOMEDIR/ollama.log 2>&1 &

        # Wait for the local Ollama HTTP API before evaluation starts.
        for i in $(seq 1 60); do
            if curl -s http://127.0.0.1:11434/api/tags > /dev/null; then
                break
            fi
            sleep 2
        done

        {cmd}

        echo "End {name}: $(date)"
    """)
