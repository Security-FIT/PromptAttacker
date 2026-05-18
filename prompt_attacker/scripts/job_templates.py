#!/usr/bin/env python3
## @file job_templates.py
#  @brief PBS job templates used by the orchestrator.
#
#  The module renders shell scripts for vLLM attacks, Ollama-backed evaluation,
#  and auxiliary workflows. Keeping PBS parameters in one place makes it easier
#  to tune GPU memory, walltime, conda activation, Ollama startup, and CUDA
#  compatibility for the cluster environment.
#
#  @author Bc. Petr Kaska
#  @date 1.2.2026
#
#  Ownership / Contribution statement:
#   - This file was designed and implemented by Bc. Petr Kaska.
#   - The PBS templates, environment setup rendering, Ollama startup handling,
#     and CUDA architecture constraints are original project infrastructure.
#   - The implementation uses standard PBS shell-script conventions.

import textwrap


def render_env_setup(env_setup):
    """@brief Render shell commands used to prepare the Python environment.

    @param env_setup Multiline shell snippet from the orchestrator config.
    @return Indented shell snippet ready for insertion into a PBS script.
    """
    return textwrap.indent(textwrap.dedent(env_setup).strip(), "        ")


def job_template(name, cmd, ollama_model, homedir=".", env_setup="module add mambaforge\nmamba activate jailbreak-exp", ollama_bin="ollama"):
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
        #PBS -l select=1:ncpus=1:ngpus=1:mem=20gb:gpu_mem=40gb:gpu_cap="sm_80,sm_90"
        #PBS -l walltime=8:00:00

        HOMEDIR={homedir}
        cd $HOMEDIR
        export PYTHONPATH="$HOMEDIR:$PYTHONPATH"

        export CUDA_VISIBLE_DEVICES=0
{render_env_setup(env_setup)}

        # Start Ollama in the background and ensure the requested model exists.
        {ollama_bin} serve > $HOMEDIR/ollama.log 2>&1 &
        {ollama_bin} pull {ollama_model}

        {cmd}

        echo "End {name}: $(date)"
    """)


def batch_template(name, cmd, ollama_model, homedir=".", env_setup="module add mambaforge\nmamba activate jailbreak-exp", ollama_bin="ollama"):
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
        #PBS -l select=1:ncpus=1:ngpus=1:mem=20gb:gpu_mem=50gb:gpu_cap="sm_80,sm_90"
        #PBS -l walltime=12:00:00

        HOMEDIR={homedir}
        cd $HOMEDIR
        export PYTHONPATH="$HOMEDIR:$PYTHONPATH"

        export CUDA_VISIBLE_DEVICES=0
{render_env_setup(env_setup)}

        {cmd}

        echo "End {name}: $(date)"
    """)


def results_eval_template(name, cmd, homedir=".", env_setup="module add mambaforge\nmamba activate jailbreak-exp", ollama_bin="ollama"):
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
        #PBS -l select=1:ncpus=1:ngpus=1:mem=50gb:gpu_mem=30gb:gpu_cap="sm_80,sm_90"
        #PBS -l walltime=15:00:00

        HOMEDIR={homedir}
        cd $HOMEDIR
        export PYTHONPATH="$HOMEDIR:$PYTHONPATH"

        export CUDA_VISIBLE_DEVICES=0
{render_env_setup(env_setup)}

        {ollama_bin} serve > $HOMEDIR/ollama.log 2>&1 &

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
