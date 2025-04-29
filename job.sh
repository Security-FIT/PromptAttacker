#!/bin/bash
#PBS -N JailBreak
#PBS -l select=1:ncpus=4:mem=4gb:scratch_local=10gb
#PBS -l walltime=1:00:00 

HOMEDIR=/storage/brno2/home/xkaska01/master/


module add mambaforge
mamba activate /storage/brno2/home/xkaska01/.conda/envs/diplomka

# test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

echo "$PBS_JOBID běží na uzlu `hostname -f`" >> $HOMEDIR/jobs_info.txt
cd $HOMEDIR

python3 2025_ICLR_PiF/PiF_CLM.py --gen_model_path models/Llama-2-13b-chat --tgt_model_path models/Llama-2-13b-chat --opt_objective ASR --interation 20 --output_dir PiF_llama3_13b_resultsMETACENTRUM