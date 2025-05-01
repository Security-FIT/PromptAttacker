# master_thesis - Jail Breaking LLMs 

### TODO
- jeste musim otestovat potom svoji obranou metodu, na tom jak ovlivni semantiku odpovedi na ne-Jailberakovy prompt

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

----------------------------------------------------------------------------------------------------------------------------
FlipAttack
<!-- https://arxiv.org/pdf/2410.02832 -->
<!-- https://github.com/yueliu1999/FlipAttack -->


Má několik parametrů, které říkají jak bude manipulováno se vstupním promptem: 
    # (I) Flip Word Order (FWO)
    # (II) Flip Chars in Word (FCW)
    # (III) Flip Chas in Sentence (FCS)
    # (IV) Fool Model Mode (FMM)

pak ma jeste 
lang_gpt = bool ... pridam systemovy prompt, kterym rikam asistentovy aby nikdy neodpovidal na nelegalni veci 
few_shot = bool ... dam modelu ukazku jak dany utok rozparsovat 
cot = bool      ... jestli chci aby popisoval krok po kroku


zatim priklad spusteni 
# python3 my_implementation/attacks/Flip/main.py --victim_llm models/Llama-2-13b-chat/ --data_path my_implementation/dataset/flip.csv

----------------------------------------------------------------------------------------------------------------------------

Sequential Break
<!-- https://arxiv.org/pdf/2411.06426v1 -->
<!-- https://anonymous.4open.science/r/JailBreakAttack-4F3B/README.md -->


----------------------------------------------------------------------------------------------------------------------------
rady od Toma:

je toho hodně co to může ovlivnit, jednak gpu, takže:
- jede ti to na gpu? Můžeš ověřit pomocí příkazu nvidia-smi, to ti ukáže kolik vram tvůj program používá, mělo by být ideálně skoro všechno.

- na PBS si musíš při spouštění úlohy specifikovat num_gpus a gpu_mem, bez toho nedostaneš žádné gpu

- kolik vramky máš k dispozici - já teď spouštím modely s 32B parametry a klidně si k tomu vezmu 70gb vramky, 18gb zabere model a zbytek je na KV cache (to právě do jisté míry určuje propustnost)

a pak se mi nejvíce osvědčilo použít knihovnu vllm přes příkazovou řádku (vllm serve), to ti spustí lokální endpoint který je kompatibilní s openai (takže k tomu přistoupíš stejně jako bys chtěl volat chat gpt přes api, třeba přes knihovnu openai)

a nejvíc cool je, že můžeš přes ssh na svém počítači forwardnout port a pak k tomu endpointu můžeš přistupovat i ze svého pc

a kdybys zvolil teda ten postup s api serverem tak doporučuju interaktivní úlohu (qsub -I) spolu s tmuxem (pokud neznáš, tak ti to umožní zachovat běžící úlohy a shell i po tom co ukončíš ssh spojení  a pak se k tomu znovu vrátit)

nevím jestli to není trochu overwhelming, tak se kdyžtak ptej a rád dovysvětlím 😅 hrozně mi trvalo, než jsem přišel na ten ideální workflow, takže ti rád ušetřím ten čas pokud možno :DD