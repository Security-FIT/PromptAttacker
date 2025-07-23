#!/bin/bash
#PDs -q default@pbs-m1.metacentrum.cz
#PBS -N JailBreak_GPU2
#PBS -l select=1:ncpus=1:ngpus=1:mem=200gb:gpu_mem=80b:scratch_local=400gb
#PBS -l walltime=20:00:00

HOMEDIR=/storage/brno2/home/xkaska01/master/


export CUDA_VISIBLE_DEVICES=0
module add mambaforge
mamba activate /storage/brno2/home/xkaska01/.conda/envs/diplomka

# test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

echo "$PBS_JOBID běží na uzlu `hostname -f`" >> $HOMEDIR/jobs_info.txt
cd $HOMEDIR
python3 -m pip install --user nltk
python3 my_implementation/run_pipeline.py --config my_implementation/config.yaml 
# python3 my_implementation/attacks/Flip/main.py --victim_llm models/Llama-2-13b-chat --temperature 0.8 --max_token 512  --flip_mode FCS --data_name advbench --begin 0 --end 100 --output_dict ./results

# python3 2025_ICLR_PiF/PiF_CLM.py --gen_model_path models/bert-large-uncased --tgt_model_path models/Llama-2-13b-chat --opt_objective ASR --interation 50 --output_dir PiF_From_Llama-2-7B_To_Llama-2-13BMETACENTRUM_GPU2
# python3 2025_ICLR_PiF/PiF_CLM.py --gen_model_path models/Llama-2-13b-chat --tgt_model_path models/Llama-2-13b-chat --opt_objective ASR --interation 20 --output_dir PiF_llama3_13b_resultsMETACENTRUM
