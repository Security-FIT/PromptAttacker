# Jail Breaking LLMs 

### TODO



Spousteni samotneho Pifu: 
<!-- python3 PiF_CLM.py --gen_model_path ../my_implementation/models/Qwen3-1.7B/ --tgt_model_path ../my_implementation/models/vicuna-7b --output_dir Pif_from -->
<!-- python3 PiF_CLM.py --gen_model_path ../my_implementation/models/Llama-2-13b-chat --tgt_model_path ../my_implementation/models/vicuna-7b --output_dir Pif_from -->

----------------------------------------------------------------------------------------------------------------------------
1✓ Cypher Attack
<!-- https://arxiv.org/pdf/2402.10601 -->
<!-- https://github.com/DivijH/jailbreak_cryptography -->


----------------------------------------------------------------------------------------------------------------------------
2✓ FlipAttack
<!-- https://arxiv.org/pdf/2410.02832 -->
<!-- https://github.com/yueliu1999/FlipAttack -->
anthropic==0.36.0
joblib==0.17.0
openai==1.51.2
pandas==1.1.3
tqdm==4.65.0

----------------------------------------------------------------------------------------------------------------------------
3 2025_ICLR_PiF
<!-- https://arxiv.org/pdf/2502.03052 -->
<!-- https://github.com/tmllab/2025_ICLR_PiF?tab=readme-ov-file -->
python PiF_CLM.py --gen_model_path ../my_implementation/models/Llama-2-7b-hf --tgt_model_path ../my_implementation/models/Llama-2-7b-hf --opt_objective ASR --interation 20 --output_dir PIF_Llama-2-7b-hf_results

python PiF_CLM.py --gen_model_path ../models/mixtral_7b --tgt_model_path ../models/mixtral_7b --opt_objective ASR --interation 20 --output_dir mixtral_7b

CUDA_VISIBLE_DEVICES=2 nohup python PiF_CLM.py --gen_model_path ../models/vicuna-13b --tgt_model_path ../models/vicuna-13b --opt_objective ASR --interation 20 --output_dir vicuna-13b

----------------------------------------------------------------------------------------------------------------------------
4✓ SQL UTOK - StructuredTransform
<!-- https://arxiv.org/pdf/2502.11853 -->
<!-- https://github.com/StructTransform/Benchmark/blob/main/easyjailbreak/datasets/harmbench_llama_SQL_subset.csv -->
----------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------
5✓ SUFIX UTOK 

<!-- https://arxiv.org/pdf/2307.15043 -->
<!-- https://github.com/llm-attacks/llm-attacks/tree/main -->

----------------------------------------------------------------------------------------------------------------------------
6✓ Sequential Break
<!-- https://arxiv.org/pdf/2411.06426v1 -->
<!-- https://anonymous.4open.science/w/JailBreakAttack-4F3B/-->


----------------------------------------------------------------------------------------------------------------------------
7✓ CitationBreak - utok cituji a tim padem llmko vse vyslepici 
<!-- https://arxiv.org/pdf/2411.11407 -->
<!-- https://github.com/YancyKahn/DarkCite -->


----------------------------------------------------------------------------------------------------------------------------
8✓ ENDLESS JAILBREAKS WITH BIJECTION LEARNING
<!-- https://arxiv.org/pdf/2410.01294v1 -->

----------------------------------------------------------------------------------------------------------------------------
9✓ Dialog_completition
<!-- https://arxiv.org/pdf/2411.06426v1 -->
<!-- https://anonymous.4open.science/w/JailBreakAttack-4F3B/-->

----------------------------------------------------------------------------------------------------------------------------
10✓  Random Search
<!-- https://arxiv.org/pdf/2404.02151 -->
<!-- https://github.com/tml-epfl/llm-adaptive-attacks -->

----------------------------------------------------------------------------------------------------------------------------
11 pair
<!-- https://arxiv.org/abs/2401.06373 -->
----------------------------------------------------------------------------------------------------------------------------
12 +-✓tap
<!-- https://arxiv.org/abs/2312.02119 -->

----------------------------------------------------------------------------------------------------------------------------
13✓ GPT4cypher
<!-- https://arxiv.org/abs/2308.06463 -->

----------------------------------------------------------------------------------------------------------------------------
14✓ A Cross-Language Investigation into Jailbreak Attacks in Large Language Models
<!-- https://arxiv.org/abs/2401.16765 -->


----------------------------------------------------------------------------------------------------------------------------
15✓ rewrite
<!-- https://arxiv.org/abs/2309.00614 -->

----------------------------------------------------------------------------------------------------------------------------
16✓ overload
<!-- https://arxiv.org/abs/2410.04190 -->
----------------------------------------------------------------------------------------------------------------------------
17✓ ica
<!-- https://arxiv.org/abs/2310.06387 -->

----------------------------------------------------------------------------------------------------------------------------
19✓ deepinception
<!-- https://arxiv.org/abs/2311.03191 -->
<!-- https://github.com/tmlr-group/DeepInception/tree/main -->
----------------------------------------------------------------------------------------------------------------------------
20✓ base
benchmark attack
----------------------------------------------------------------------------------------------------------------------------
21✓ Art prompt
<!-- https://arxiv.org/abs/2402.11753 -->
<!-- https://github.com/uw-nsl/ArtPrompt -->

----------------------------------------------------------------------------------------------------------------------------
22✓ Renellm
<!-- https://arxiv.org/abs/2311.08268 -->

----------------------------------------------------------------------------------------------------------------------------
23✓ COLD
<!-- https://arxiv.org/abs/2402.08679 -->
<!-- https://github.com/Yu-Fangxu/COLD-Attack -->

----------------------------------------------------------------------------------------------------------------------------
24 autodan 
<!-- https://arxiv.org/abs/2310.04451 -->
<!-- https://github.com/SheltonLiu-N/AutoDAN/blob/main/utils/opt_utils.py#L176 -->
----------------------------------------------------------------------------------------------------------------------------
25✓ Past tense
<!-- https://arxiv.org/abs/2407.11969 -->
<!-- https://github.com/tml-epfl/llm-past-tense/blob/main/main.py -->

----------------------------------------------------------------------------------------------------------------------------
26✓ Chameleon
<!-- https://arxiv.org/abs/2402.16717 -->
<!-- https://github.com/huizhang-L/CodeChameleon -->

----------------------------------------------------------------------------------------------------------------------------