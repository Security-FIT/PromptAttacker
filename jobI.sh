qsub -I -l walltime=8:0:0 -q default@pbs-m1.metacentrum.cz -l select=1:ncpus=1:ngpus=1:mem=200gb:gpu_mem=60gb:scratch_local=400gb
module add mambaforge
mamba activate /storage/brno2/home/xkaska01/.conda/envs/diplomka
cd /storage/brno2/home/xkaska01/master

export CUDA_VISIBLE_DEVICES=0