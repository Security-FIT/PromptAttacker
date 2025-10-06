#!/bin/bash
#PDs -q gpu_long@pbs-m1.metacentrum.cz
#PBS -N job5_GPU2
#PBS -l select=1:ncpus=1:ngpus=1:mem=5gb:gpu_mem=45gb
#PBS -l walltime=96:00:00

HOMEDIR=/storage/brno2/home/xkaska01/master/


export CUDA_VISIBLE_DEVICES=0
module add mambaforge
mamba activate /storage/brno2/home/xkaska01/.conda/envs/diplomka

#na zapnuti ollama serveru

# /storage/brno2/home/xkaska01/test/bin/ollama pull deepseek-r1:32b
/storage/brno2/home/xkaska01/test/bin/ollama pull qwen2.5:7b
# /storage/brno2/home/xkaska01/test/bin/ollama pull llama3.1:70b
# /storage/brno2/home/xkaska01/test/bin/ollama pull falcon3:10b
# /storage/brno2/home/xkaska01/test/bin/ollama pull gemma3:27b
# /storage/brno2/home/xkaska01/test/bin/ollama pull qwen3:32b
# /storage/brno2/home/xkaska01/test/bin/ollama pull yi:34b
/storage/brno2/home/xkaska01/test/bin/ollama pull internlm2:7b
# /storage/brno2/home/xkaska01/test/bin/ollama pull command-r:35b
/storage/brno2/home/xkaska01/test/bin/ollama serve > $HOMEDIR/ollama.log 2>&1 &

echo "$PBS_JOBID běží na uzlu `hostname -f`" >> $HOMEDIR/jobs_info.txt
cd $HOMEDIR
python3 -m pip install --user nltk
python3 my_implementation/run_pipeline.py --config my_implementation/config5.yaml 
# python3 my_implementation/attacks/Flip/main.py --victim_llm models/Llama-2-13b-chat --temperature 0.8 --max_token 512  --flip_mode FCS --data_name advbench --begin 0 --end 100 --output_dict ./results
# python3 2025_ICLR_PiF/PiF_CLM.py --gen_model_path models/bert-large-uncased --tgt_model_path models/Llama-2-13b-chat --opt_objective ASR --interation 50 --output_dir PiF_From_Llama-2-7B_To_Llama-2-13BMETACENTRUM_GPU2
# python3 2025_ICLR_PiF/PiF_CLM.py --gen_model_path models/Llama-2-13b-chat --tgt_model_path models/Llama-2-13b-chat --opt_objective ASR --interation 20 --output_dir PiF_llama3_13b_resultsMETACENTRUM


# prikazy pro spusteni ollamy z my_implementation
# ./../../test/bin/ollama serve &