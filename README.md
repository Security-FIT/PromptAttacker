# master_thesis - Jail Breaking LLMs 

spousteni skriptu na GPU: CUDA_VISIBLE_DEVICES=1,2 python tvuj_skript.pyp


Dalsi modely
VIcuna 13b - https://huggingface.co/lmsys/vicuna-13b-v1.5
mixtral 7b - https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
llama 2 8b
llama 3 13b


----------------------------------------------------------------------------------------------------------------------------
2025_ICLR_PiF
<!-- https://arxiv.org/pdf/2502.03052 -->
<!-- https://github.com/tmllab/2025_ICLR_PiF?tab=readme-ov-file -->
python PiF_CLM.py --gen_model_path ../models/Llama-2-13b-chat --tgt_model_path ../models/Llama-2-13b-chat --opt_objective ASR --interation 20 --output_dir PiF_llama3_13b_results

python PiF_CLM.py --gen_model_path ../models/mixtral_7b --tgt_model_path ../models/mixtral_7b --opt_objective ASR --interation 20 --output_dir mixtral_7b

CUDA_VISIBLE_DEVICES=2 nohup python PiF_CLM.py --gen_model_path ../models/vicuna-13b --tgt_model_path ../models/vicuna-13b --opt_objective ASR --interation 20 --output_dir vicuna-13b


stahnute modely k tomuto Pifu...
mixtral 7b
llama 2 8b
llama 3 13b

----------------------------------------------------------------------------------------------------------------------------

FlipAttack
<!-- https://arxiv.org/pdf/2410.02832 -->
<!-- https://github.com/yueliu1999/FlipAttack -->
mixtral 7b
llama 3 13b

----------------------------------------------------------------------------------------------------------------------------

Sequential Break
<!-- https://arxiv.org/pdf/2411.06426v1 -->
<!-- https://anonymous.4open.science/r/JailBreakAttack-4F3B/README.md -->