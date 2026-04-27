# Jail Breaking LLMs 
# Jail Breaking LLMs — Project Overview

Tento repozitář obsahuje nástroje pro generování, spouštění a vyhodnocení „attack“ scénářů proti velkým jazykovým modelům (LLM). Cílem je experimentálně zkoumat zranitelnosti modelů, testovat obranné přístupy a sbírat metriky kvality odpovědí.

Repo je rozdělené na několik subsystémů:

- `my_implementation/attacks/` — implementace jednotlivých útoků (cypher, flip, pif, …).
- `my_implementation/defense/` — jednoduché obranné moduly a nástroje pro přepis promptů.
- `my_implementation/evaluate/` — skripty pro automatické hodnocení odpovědí.
- `my_implementation/results/` — výstupy (útoky, prompt datasets, odpovědi, statistiky).
- Orchestrátory: `my_implementation/run_orchestrator.py` (unifikovaný CLI) a starší `my_implementation/run.py` (legacy job generator).

Soubor s centrálním nastavením pro orchestrátor je [my_implementation/config_orchestrator.yaml](my_implementation/config_orchestrator.yaml).

**Důležité:** používejte virtuální prostředí s potřebnými závislostmi; některé části projektu předpokládají běžící Ollama REST API pro inference (viz sekce "Rychlá doporučení").

**Rychlé shrnutí rozdílu `run_orchestrator.py` vs `run.py`**
- `run_orchestrator.py` je moderní, strukturovaný orchestrátor s podpůrnými subcommandy: `attack`, `infer`, `defend-dataset`, `evaluate`, `train-defense` a `pipeline`. Konfigurace se čte z `config_orchestrator.yaml`.
- `run.py` je starší skript specifický pro váš experimentální workflow: obsahuje hodně hardcodovaných nastavení (sady modelů, job templates) a vlastní režimy jako `--only-attack`, `--only-attack-batch`, `--fix` apod.
- Funkčně se části překrývají (oba vytvářejí PBS joby, oba spouští inference), ale nejsou zaměnitelné bez úprav konfigurací nebo kódu. Pokud chcete konzistentní, přenosný a rozšiřitelný nástroj, preferujte `run_orchestrator.py`.

**Krátká odpověď na otázku:** Ano, `run_orchestrator.py` pokrývá stejnou základní funkcionalitu jako některé režimy `run.py` (vytváření jobů a batch inference), ale je přehlednější a konfigurovatelný přes `config_orchestrator.yaml`. `run.py` obsahuje navíc mnoho experimentálních/hardcoded kroků a seznamů modelů.

---

**Požadavky**

- Python 3.9+ (doporučeno 3.10/3.11)
- Doporučené balíčky: `pyyaml`, `requests` (viz `requirements.txt` pokud projekt přidáte)
- Doporučené: Ollama pro lokální inference (pokud používáte `use_ollama: true` v konfiguraci)

---

**Rychlý start (Quickstart)**

1. Vytvořte/aktivujte virtuální prostředí a nainstalujte závislosti:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pyyaml requests
```

2. Upravit konfiguraci: [my_implementation/config_orchestrator.yaml](my_implementation/config_orchestrator.yaml)

3. Spustit orchestrátor (příklad: pipeline v interaktivním módu):

```bash
python3 my_implementation/run_orchestrator.py pipeline --mode interactive --config my_implementation/config_orchestrator.yaml
```

4. Spustit inferenci v batch módu (zrychlí inference přes PBS joby):

```bash
python3 my_implementation/run_orchestrator.py infer --mode batch --config my_implementation/config_orchestrator.yaml
```

Poznámka: `--mode batch` vygeneruje PBS skripty do složky `jobs_orchestrator` (výchozí) a při `--submit` je automaticky odešle (`qsub`).

---

**Konfigurace (stručně)**

Hlavní konfigurační soubor: [my_implementation/config_orchestrator.yaml](my_implementation/config_orchestrator.yaml). Důležitá pole:

- `model.victim_llm` — šablona nebo cesta k per-victim modelu (podporuje `{model}` substitution).
- `model.use_ollama` — jestli se volá lokální Ollama API.
- `attack.dataset` — vstupní CSV/JSON pro generování útoků.
- `attack.out_dir` — kde se ukládají výstupy útoků.
- `infer.input` — složka s prompt JSONy pro inferenci.
- `infer.out_dir` — kam se uloží odpovědi.
- `infer.defense` — `none`, `rallm`, `llamaguard`, `safeguard`, `ea`, `rule`.
- `infer.batch_size` — počet promptů zpracovaných v jedné dávce (větší = efektivnější pokud backend podporuje batchy).
- `pipeline.*` — parametry pro celou pipeline (attack -> infer -> evaluate).

Upravte hodnoty přímo v konfiguraci nebo přepište CLI argumenty (např. `--batch-size`).

---

**Jak `run_orchestrator.py` používat — praktické příklady**

- Generování útoků (lokálně, interaktivně):

```bash
python3 my_implementation/run_orchestrator.py attack --config my_implementation/config_orchestrator.yaml --methods works
```

- Inferenční běh nad existujícími prompt JSONy (rychlejší: batch joby):

```bash
# vytvoří skripty do jobs_orchestrator/ a neodešle
python3 my_implementation/run_orchestrator.py infer --mode batch --config my_implementation/config_orchestrator.yaml

# vytvoří skripty a odešle qsub
python3 my_implementation/run_orchestrator.py infer --mode batch --submit --config my_implementation/config_orchestrator.yaml
```

- Spustit celou pipeline (attack -> infer -> evaluate):

```bash
python3 my_implementation/run_orchestrator.py pipeline --mode interactive --config my_implementation/config_orchestrator.yaml
```

Tip: pro rychlou inference na lokálním stroji bez PBS použijte `--mode interactive` a nastavte `--batch-size` vyšší hodnotu (pokud backend podporuje batch calls). Pokud máte PBS cluster, preferujte `--mode batch --submit`.

---

**Doporučení pro rychlejší inference**

- Použijte batch režim: buď CLI `--mode batch` (vytvoří joby) nebo zvýšte `infer.batch_size`.
- Deaktivujte verbose/interactive volání: v `config_orchestrator.yaml` nastavte `infer.interactive_calls: false` nebo nepředávejte `--interactive-calls`.
- Pokud používáte Ollama, volte režim `use_ollama: true` a ujistěte se, že server běží a modely jsou přednahrané (`ollama pull`).
- Pro velké množiny promptů rozdělte dataset a spouštějte paralelně přes PBS (více jobů). `run_orchestrator.py` automaticky generuje PBS skripty při `--mode batch`.

---

**Porovnání — kdy použít který skript**

- `my_implementation/run_orchestrator.py`: preferovaný orchestrátor pro nové experimenty, konzistentní CLI, podporuje `pipeline` a defense options. Dobré pro reprodukovatelnost a CI.
- `my_implementation/run.py`: legacy skript s mnoha experimentálními hacky a vlastním job managementem. Použijte ho pokud spoléháte na konkrétní hardcoded seznamy modelů nebo na existující custom job templates v `run.py`.

Pokud chcete sjednotit workflow, doporučuji migrovat logiku z `run.py` do `run_orchestrator.py` jen pro ty režimy, které pravidelně používáte.

---

**Přispívání a další kroky**

- Přidejte `requirements.txt` nebo `pyproject.toml` pro jednodušší instalaci závislostí.
- Přidejte jednoduché testy (smoke tests) pro `run_orchestrator` subcommandy.
- Pokud chcete, můžu:
	- doplnit `requirements.txt`
	- přidat příklady PBS jobů do `docs/`
	- upravit `run_orchestrator.py` tak, aby překryl chybějící chování z `run.py`

---

Autor: tým experimentů — upravte podle potřeby

**CLI Reference: `run_orchestrator.py`**

- **`--config <path>`:** Cesta ke konfiguračnímu YAML souboru (výchozí `config_orchestrator.yaml`).
- **`--evaluate` :** Vygeneruje eval job skripty pro model(y) uvedené v konfiguraci. S `--interactive` spustí evaluaci přímo v terminálu.
- **`--attack-single` :** Vytvoří job skripty (nebo spustí interaktivně) pro jednotlivé útoky nad jedním cílovým modelem (`target_model` / `ollama_model`).
- **`--attack-batch` :** Pro každý soubor v `dataset_to_attack_path` vytvoří job skript volající `only_attack_batch.py` (nebo spustí interaktivně). Iteruje přes `target_models` / `ollama_models`.
- **`--defense {rallm,llamaguard,safeguard}` :** Vytvoří/ spustí obranné joby (volá `only_defense_batch.py`) pro zvolený typ obrany nad datasetem.
- **`--run-pipeline` :** Spustí `run_pipeline.py` pro celý pipeline (attack -> infer -> evaluate). Pokud chybí `--interactive`, vytvoří jeden job skript `job_run_pipeline.sh` v adresáři `results/<results_dir>/jobs`.
- **`--pipeline-out <dir>` :** Přepíše `results_dir` pro výstupy pipeline; používané společně s `--run-pipeline`.
- **`--interactive` :** Pokud je nastaveno, akce se spustí přímo v terminálu pomocí `subprocess.run(...)` místo vytváření job skriptů.

Konfigurační klíče důležité pro chování:
- **`dry_run` (v YAML):** pokud `true`, joby se vygenerují, ale neodesílají se (`qsub` nebude voláno).
- **`use_ollama` :** `true|false` — zda volat lokální Ollama REST API (localhost:11434) nebo použít lokální model path.
- **`local_model_path` / `victim_llm` :** cesta nebo šablona k lokálnímu modelu; podporuje `{model}` substituci.
- **`target_model` / `ollama_model` :** model používaný pro single-run.
- **`target_models` / `ollama_models` :** seznam modelů pro batch iteraci.
- **`results_dir` :** hlavní výstupní adresář; job skripty ukládá `.../jobs`.
- **`dataset_to_attack_path` :** složka s .json soubory používanými pro batch útoky.

Příklady (v `my_implementation` adresáři):

1) Vytvoření job skriptů pro všechny útoky (dry-run v configu):

```bash
python3 run_orchestrator.py --config config_orchestrator.yaml --attack-batch
```

2) Interaktivní evaluace pro testovací model (spustí se přímo):

```bash
python3 run_orchestrator.py --config config_orchestrator_test.yaml --evaluate --interactive
```

3) Spustit pipeline interaktivně a uložit výstupy jinam:

```bash
python3 run_orchestrator.py --config config_orchestrator.yaml --run-pipeline --pipeline-out results/oponent_pipeline --interactive
```

4) Vytvořit jeden job, který spustí pipeline (bez interaktivního běhu):

```bash
python3 run_orchestrator.py --config config_orchestrator.yaml --run-pipeline --pipeline-out results/oponent_pipeline
```

Kde hledat vygenerované job skripty:
- `results/<results_dir>/jobs/` — všechny vytvořené skripty (např. `job_onlyattack_*.sh`, `job_run_pipeline.sh`).

Další skripty a krátký přehled:
- `run.py` — starší/legacy job generator (obsahuje `--only-attack`, `--only-attack-batch` apod.).
- `run_pipeline.py` — vykonává pipeline podle `config_pipeline_temp.yaml` nebo jiného `--config_file`.
- `only_attack.py`, `only_attack_batch.py`, `only_defense_batch.py` — malé utility, které orchestrátor volá buď interaktivně, nebo v job skriptech.

Pokud chceš, doplním README o příklady `qsub` headerů, ukázky obsahu vygenerovaných `job_*.sh` skriptů nebo automatický smoke-test section.
