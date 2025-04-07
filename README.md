# master_thesis - Jail Breaking LLMs 



<!-- https://arxiv.org/pdf/2502.03052 -->
<!-- https://github.com/tmllab/2025_ICLR_PiF?tab=readme-ov-file -->
python PiF_CLM.py --gen_model_path ../models/Llama-3-13b-chat --tgt_model_path ../models/Llama-3-13b-chat --opt_objective ASR --interation 20 --output_dir PiF_llama3_13b_results

spousteni skriptu na GPU: CUDA_VISIBLE_DEVICES=1,2 python tvuj_skript.pyp