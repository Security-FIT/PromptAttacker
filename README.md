# master_thesis - Jail Breaking LLMs 

### TODO
- jeste musim otestovat potom svoji obranou metodu, na tom jak ovlivni semantiku odpovedi na ne-Jailberakovy prompt

spousteni skriptu na GPU: CUDA_VISIBLE_DEVICES=1,2 python tvuj_skript.pyp


Dalsi modely - viz models/ a příslušná stránka na HG (všechny modely stažené z HF)
VIcuna 13b - https://huggingface.co/lmsys/vicuna-13b-v1.5
mixtral 7b - https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1



----------------------------------------------------------------------------------------------------------------------------
1 Cypher Attack
<!-- https://arxiv.org/pdf/2402.10601 -->
<!-- https://github.com/DivijH/jailbreak_cryptography -->


Tento utok jsem musel celý implementovat podle sebe v článku je několik metod šifrování vstupního textu, ale já jsem si zvolil pouze tu nejúspěšnější. (Tedy metodu, která měla největší úspěch Jailbreakovat model)

Dale jsem doimplementoval pomer v jakem se slova nahrazuji. Tedy delsi vety nahradi vice slov a kratsi mene slov.
-- TOTO BUDU JESTE MUSET OVERIT EXPERIMENTY A POPRIPADE ZMENIT !!!!!!


----------------------------------------------------------------------------------------------------------------------------
2 FlipAttack
<!-- https://arxiv.org/pdf/2410.02832 -->
<!-- https://github.com/yueliu1999/FlipAttack -->
anthropic==0.36.0
joblib==0.17.0
openai==1.51.2
pandas==1.1.3
tqdm==4.65.0


Má několik parametrů, které říkají jak bude manipulováno se vstupním promptem: 
    # (I) Flip Word Order (FWO)
    # (II) Flip Chars in Word (FCW)
    # (III) Flip Chas in Sentence (FCS)
    # (IV) Fool Model Mode (FMM)

pak ma jeste 
lang_gpt = bool ... pridam systemovy prompt, kterym rikam asistentovy aby nikdy neodpovidal na nelegalni veci 
few_shot = bool ... dam modelu ukazku jak dany utok rozparsovat 
cot = bool      ... jestli chci aby popisoval krok po kroku


----------------------------------------------------------------------------------------------------------------------------
3 2025_ICLR_PiF
<!-- https://arxiv.org/pdf/2502.03052 -->
<!-- https://github.com/tmllab/2025_ICLR_PiF?tab=readme-ov-file -->
python PiF_CLM.py --gen_model_path ../my_implementation/models/Llama-2-13b-chat --tgt_model_path ../my_implementation/models/Llama-2-13b-chat --opt_objective ASR --interation 20 --output_dir PiF_llama3_13b_results

python PiF_CLM.py --gen_model_path ../models/mixtral_7b --tgt_model_path ../models/mixtral_7b --opt_objective ASR --interation 20 --output_dir mixtral_7b

CUDA_VISIBLE_DEVICES=2 nohup python PiF_CLM.py --gen_model_path ../models/vicuna-13b --tgt_model_path ../models/vicuna-13b --opt_objective ASR --interation 20 --output_dir vicuna-13b

----------------------------------------------------------------------------------------------------------------------------
4 SQL UTOK - StructuredTransform
<!-- https://arxiv.org/pdf/2502.11853 -->
<!-- https://github.com/StructTransform/Benchmark/blob/main/easyjailbreak/datasets/harmbench_llama_SQL_subset.csv -->
----------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------
5 SUFIX UTOK - modifikuji sufixy a utocim s nimi na llmka
generuji sufixy nez mi llmko odpovi na prompt - Teze je ze llmko se jailbreakne pokud na nej aplikuji určitý sufix a tento sufix pak bude fungovat u všech promptů

<!-- https://arxiv.org/pdf/2307.15043 -->
<!-- https://github.com/llm-attacks/llm-attacks/tree/main -->

!!!! Tento utok jeste dodelam az budu mit GPT API, protoze si musim vytvorit takovy sufix aby se dokazal dostat do jinych llmek, GPT API mi bude sufix modifikovat do te doby nez Jailbreaknu target model, a potom podle Faktů z tohoto Paperu by daný sufix měl zabýrat pro se všemi prompty pro daný model. Tedy stačí mi najít jeden sufix a projdou mi všechny Jailbreaky, toto tedy udělám pro všechny modely se kterými budu chtít experimentovat.


Tedy tento utok je zatím WORK IN PROGRES A VELIKÉ TODOOO
----------------------------------------------------------------------------------------------------------------------------
6 Sequential Break
<!-- https://arxiv.org/pdf/2411.06426v1 -->
<!-- https://anonymous.4open.science/w/JailBreakAttack-4F3B/-->


----------------------------------------------------------------------------------------------------------------------------
7 CitationBreak - utok cituji a tim padem llmko vse vyslepici 
<!-- https://arxiv.org/pdf/2411.11407 -->
<!-- https://github.com/YancyKahn/DarkCite -->


----------------------------------------------------------------------------------------------------------------------------
8 ENDLESS JAILBREAKS WITH BIJECTION LEARNING
<!-- https://arxiv.org/pdf/2410.01294v1 -->

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