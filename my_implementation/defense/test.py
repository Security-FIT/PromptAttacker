#!/usr/bin/env python3
import json, random, re, requests, os, glob

# ========= HARD-CODED NASTAVENÍ ========= #
INPUT_DIR  = "/storage/brno2/home/xkaska01/master/my_implementation/results/without_defense/DATASET_C_internlm2:7b"
OUTPUT_DIR = "/storage/brno2/home/xkaska01/master/my_implementation/results/with_defense/DATASET_C_internlm2:7b"

RULE_PATH  = "/storage/brno2/home/xkaska01/master/my_implementation/defense/defense_rule.json"

VOCAB_PATH = "/storage/brno2/home/xkaska01/master/my_implementation/defense/models_vocabularies/internlm2_5-7b-chat-1m_vocab.txt"
# Volitelný vocab
USE_VOCAB  = False  # True = použít vocab; False = jen symboly/kombinace dle pravidla

ENDPOINT   = "http://localhost:11434/api/generate"
MODEL      = "internlm/internlm2.5:latest"

# Symbolický noise pool
SYMBOL_POOL = list(';+"£$%^&*()<>?:}{')

TIMEOUT    = 120
# ======================================== #

_NON_ALNUM = re.compile(r'[^0-9A-Za-z\u00C0-\u024f]')
PUNCT_BETWEEN = re.compile(r'[,.!?;:]+')


def load_rule_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    return j.get("rule_tree", j)


def load_vocab_tokens(vocab_path: str):
    toks = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            _, tok_str = parts
            tok_str = tok_str.strip()
            try:
                tok = json.loads(tok_str)
                if isinstance(tok, str):
                    toks.append(tok)
            except Exception:
                toks.append(tok_str.strip('"'))
    return toks


def build_token_pool(tokens, min_alpha_len=4):
    pool = set()
    for t in tokens:
        if not t:
            continue
        tok = t
        if tok.startswith("▁") and len(tok) > 1:
            pool.add(tok); continue
        if _NON_ALNUM.search(tok):
            pool.add(tok); continue
        if tok.isalpha() and len(tok) >= min_alpha_len:
            pool.add(tok); continue
    return sorted(pool)


def render_token_for_insert(tok: str) -> str:
    return tok[1:] if tok.startswith("▁") else tok


def _snap_to_internal_boundary(tokens, j):
    n = len(tokens)
    if n <= 2:
        return max(0, min(j, n-1))
    j = max(1, min(j, n-2))
    best, best_dist = j, n + 1
    for i in range(1, n-1):
        if PUNCT_BETWEEN.search(tokens[i-1]):
            d = abs(i - j)
            if d < best_dist:
                best, best_dist = i, d
    return best


def _enforce_dispersion(indices, d_min, n):
    if not indices:
        return []
    indices = sorted(set(max(0, min(i, n-1)) for i in indices))
    kept, last = [], -10**9
    for i in indices:
        if i - last >= d_min:
            kept.append(i); last = i
    return kept


def _insert_at_index(tokens, idx, piece, mode):
    if mode == "wrap":
        pre, suf = piece
        tokens[idx] = f"{pre}{tokens[idx]}{suf}"
    elif mode == "prefix":
        tokens[idx] = f"{piece}{tokens[idx]}"
    elif mode == "suffix":
        tokens[idx] = f"{tokens[idx]}{piece}"
    return tokens


def _select_token_pool(rule: dict, vocab_pool, symbol_pool):
    subset = str(rule.get("insertion", {}).get("vocab_subset", "")).lower()
    if subset == "symbols_only":
        return list(symbol_pool)
    if subset == "tokens":
        return list(vocab_pool) if vocab_pool else list(symbol_pool)
    if vocab_pool:
        return list(vocab_pool) + list(symbol_pool)
    return list(symbol_pool)


def apply_rule_to_sentence(sentence: str, rule: dict, vocab_pool, symbol_pool):
    tokens = sentence.split()
    n = len(tokens)
    if n == 0:
        return sentence

    pool = _select_token_pool(rule, vocab_pool, symbol_pool)
    if not pool:
        pool = ["▁xyz", "▁abc", "▁qwe", "▁rnd"]

    delta = float(rule["anchors"].get("delta", 0.05))
    qs = [q for q in rule["anchors"].get("quantiles", []) if delta < q < 1.0 - delta]
    anchor_idx = [max(0, min(int(q * n), n - 1)) for q in qs]

    d_min = int(rule["dispersion"].get("d_min", 1))
    anchor_idx = _enforce_dispersion(anchor_idx, d_min, n)

    if not anchor_idx and n >= 3:
        anchor_idx = [_snap_to_internal_boundary(tokens, n // 2)]

    snapped = [_snap_to_internal_boundary(tokens, i) for i in anchor_idx]

    rho = float(rule["insertion"].get("budget_rho", 0.01))
    budget = max(1, int(rho * n))  # rozpočet v „k-blocích“

    m = len(snapped)
    if m == 0:
        return " ".join(tokens)
    per = [budget // m] * m
    for t in range(budget % m):
        per[t] += 1

    k_min = int(rule["insertion"].get("k_min", 1))
    k_max = int(rule["insertion"].get("k_max", max(k_min, 1)))
    mode = rule["insertion"].get("mode", "wrap")

    def _make_payload(k):
        pre = "".join(render_token_for_insert(random.choice(pool)) for _ in range(k))
        suf = "".join(render_token_for_insert(random.choice(pool)) for _ in range(k))
        return pre, suf

    for idx, alloc in zip(snapped, per):
        remaining = alloc
        while remaining > 0:
            k = max(k_min, min(k_max, remaining))
            if mode == "wrap":
                pre, suf = _make_payload(k)
                tokens = _insert_at_index(tokens, idx, (pre, suf), mode)
            elif mode == "prefix":
                piece = "".join(render_token_for_insert(random.choice(pool)) for _ in range(k))
                tokens = _insert_at_index(tokens, idx, piece, mode)
            else:
                piece = "".join(render_token_for_insert(random.choice(pool)) for _ in range(k))
                tokens = _insert_at_index(tokens, idx, piece, mode)
            remaining -= k

    return " ".join(tokens)


def _extract_text_from_response(resp_json) -> str:
    if isinstance(resp_json, dict):
        if isinstance(resp_json.get("response"), str):
            return resp_json["response"].strip()
        if "choices" in resp_json and isinstance(resp_json["choices"], list) and resp_json["choices"]:
            ch0 = resp_json["choices"][0]
            if isinstance(ch0, dict):
                if isinstance(ch0.get("text"), str):
                    return ch0["text"].strip()
                if isinstance(ch0.get("message"), dict) and isinstance(ch0["message"].get("content"), str):
                    return ch0["message"]["content"].strip()
    try:
        return json.dumps(resp_json, ensure_ascii=False)
    except Exception:
        return str(resp_json)


def call_model(prompt: str):
    payload = {"model": MODEL, "prompt": prompt, "max_tokens": 512, "temperature": 0.7, "stream": False}
    r = requests.post(ENDPOINT, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def output_name_from_input(input_filename: str) -> str:
    """
    '2_flip.json' -> 'flip.json'
    '11_pair.json' -> 'pair.json'
    'base.json' -> 'base.json' (bez změny)
    """
    base = os.path.basename(input_filename)
    return re.sub(r'^\d+[_-]?', '', base)


def process_file(path_in: str, rule, vocab_pool):
    # načti JSON pole
    with open(path_in, "r", encoding="utf-8") as f:
        try:
            items = json.load(f)
        except Exception as e:
            print(f"[WARN] Skipping {path_in}: JSON parse error: {e}")
            return

    if not isinstance(items, list):
        print(f"[WARN] Skipping {path_in}: not a JSON array")
        return

    out_items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        _id = item.get("id", len(out_items))
        original_prompt = item.get("original_prompt", "")
        prompt_in = item.get("prompt", original_prompt or "")

        # aplikuj obranu na 'prompt'
        transformed_prompt = apply_rule_to_sentence(prompt_in, rule, vocab_pool, SYMBOL_POOL)

        # zavolej model a PŘEPIŠ response
        try:
            resp_json = call_model(transformed_prompt)
            response_text = _extract_text_from_response(resp_json)
        except Exception as e:
            response_text = f"[ERROR] API call failed: {e}"

        out_items.append({
            "id": _id,
            "original_prompt": original_prompt,   # beze změny
            "prompt": transformed_prompt,         # upravený prompt
            "response": response_text             # nová odpověď
        })

    # ulož
    ensure_dir(OUTPUT_DIR)
    out_name = output_name_from_input(path_in)
    out_path = os.path.join(OUTPUT_DIR, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_items, f, ensure_ascii=False, indent=2)
    print(f"[OK] {path_in} -> {out_path} ({len(out_items)} items)")


if __name__ == "__main__":
    rule = load_rule_json(RULE_PATH)

    vocab_pool = []
    if USE_VOCAB and VOCAB_PATH:
        try:
            toks = load_vocab_tokens(VOCAB_PATH)
            vocab_pool = build_token_pool(toks, min_alpha_len=4)
        except Exception as e:
            print(f"[WARN] Vocab load failed: {e}. Proceeding without vocab.")
            vocab_pool = []

    # projdi všechny .json soubory v INPUT_DIR
    pattern = os.path.join(INPUT_DIR, "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[WARN] No JSON files found in {INPUT_DIR}")
    for path_in in files:
        process_file(path_in, rule, vocab_pool)
