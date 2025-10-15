import os, re, json, csv, glob, sys, time
from typing import List, Any, Optional
import requests

# --- CONFIG ---
MODEL_NAME = "command-r:35b"  # POZOR na pozici ':'
INPUT_DIR = f"/storage/brno2/home/xkaska01/master/my_implementation/results/DATASET_FULL_{MODEL_NAME}/"
JUDGE_MODEL = "gemma3:12b"
OLLAMA_HOST = "http://localhost:11434"
GENERATE_PATH = "/api/generate"

EVAL_INSTRUCTION = """... (ponech tvůj původní prompt) ..."""
EVAL_TEMPLATE = """PROMPT:
{prompt}

ANSWER:
{answer}

Now output ONLY a single integer from 0 to 10:"""

INT_RE = re.compile(r"\b([0-9]|10)\b")

def call_ollama(host: str, model: str, prompt: str, max_retries: int = 3, timeout: int = 1200) -> str:
    url = host.rstrip("/") + GENERATE_PATH
    payload = {"model": model, "prompt": EVAL_INSTRUCTION + "\n\n" + prompt, "stream": False, "options": {"temperature": 0.0}}
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            last_err = e
            time.sleep(min(2 * attempt, 6))
    raise RuntimeError(f"Ollama request failed: {last_err}")

def extract_score(raw: Any) -> Optional[int]:
    if isinstance(raw, str):
        m = INT_RE.search(raw.strip())
        if m:
            v = int(m.group(1))
            if 0 <= v <= 10:
                return v
    return None

def score_item(prompt_text: str, answer_text: str) -> int:
    raw = call_ollama(OLLAMA_HOST, JUDGE_MODEL, EVAL_TEMPLATE.format(prompt=prompt_text, answer=answer_text))
    s = extract_score(raw)
    if s is None:
        raw2 = call_ollama(OLLAMA_HOST, JUDGE_MODEL, "Return only one integer (0-10) for the previous evaluation.")
        s = extract_score(raw2)
    return 0 if s is None else s

def infer_attack_type(filename: str) -> str:
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    m = re.match(r"_?\d+[ _-]*(.*)", name)
    if m and m.group(1):
        return m.group(1).strip().replace(" ", "_")
    return re.sub(r"^[^A-Za-z]+", "", name).replace(" ", "_") or name

def out_csv_path(model_name: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))  # adresář, kde leží skript
    safe = model_name.replace("/", "_")
    return os.path.join(script_dir, f"{safe}.csv")

def main():
    start_time = time.time()

    if not os.path.isdir(INPUT_DIR):
        raise SystemExit(f"INPUT_DIR není adresář: {INPUT_DIR}")

    rows = []       # [(attack_type, [scores...])]
    max_len = 0

    for fp in sorted(glob.glob(os.path.join(INPUT_DIR, "*.json"))):
        attack_type = infer_attack_type(fp)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                items = json.load(f)
            if not isinstance(items, list):
                print(f"[WARN] Přeskakuji (není list): {fp}", file=sys.stderr)
                continue
        except Exception as e:
            print(f"[WARN] Přeskakuji (nelze načíst): {fp}: {e}", file=sys.stderr)
            continue

        scores: List[int] = []
        for it in items:
            prompt_text = ((it.get("original_prompt") or "").strip() or (it.get("prompt") or ""))
            answer_text = (it.get("response") or "").strip()
            if not prompt_text or not answer_text:
                scores.append(0)
            else:
                try:
                    scores.append(score_item(prompt_text, answer_text))
                except Exception as e:
                    print(f"[ERROR] Scoring selhal pro id={it.get('id')} v {attack_type}: {e}", file=sys.stderr)
                    scores.append(0)

        rows.append((attack_type, scores))
        max_len = max(max_len, len(scores))
        print(f"[OK] {attack_type}: {len(scores)} položek")

    # Zápis širokého CSV do stejné složky, kde leží skript
    out_csv = out_csv_path(MODEL_NAME)
    header = ["attack_type"] + [f"score_{i}" for i in range(max_len)]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for attack_type, scores in rows:
            w.writerow([attack_type] + scores + [""] * (max_len - len(scores)))

    print(f"[OK] CSV zapsáno → {out_csv}")
    end_time = time.time()

    total_sec = int(end_time - start_time)
    mins = total_sec // 60
    secs = total_sec % 60
    print(f"[OK] Hotovo za {mins}m {secs}s")

if __name__ == "__main__":
    main()
