#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Načte CSV: goal, translation_of_goal[, citation], target
→ pro každý goal vygeneruje 1 LLM citaci a VŽDY přepíše sloupec 'citation'
→ zajistí pořadí: goal, translation_of_goal, citation, target, ...

Model: google/flan-t5-small (funguje i na CPU)
"""

import re
import hashlib
from typing import List, Dict, Iterable

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False

# ========= CESTY =========
IN_CSV  = "/storage/brno2/home/xkaska01/master/my_implementation/dataset/base_dataset_ZH_citations.csv"
OUT_CSV = "/storage/brno2/home/xkaska01/master/my_implementation/dataset/base_dataset_ZH_citations_llm.csv"
# =========================

MODEL_NAME = "google/flan-t5-small"
MAX_NEW_TOKENS = 96
NUM_BEAMS = 4
BATCH_SIZE = 8

CITATION_MODES = ["journal", "conference", "book", "wikipedia"]

SENSITIVE = {
    "bomb","explosive","weapon","malware","ransomware","ddos","exploit","poison","assassination","drug","counterfeit",
    "harm","kill","shoot","grenade","napalm","meth","cocaine","heroin","bruteforce","lockpick","terror","hack",
    "bypass","jailbreak","credit card","skimmer","phishing","keylogger","breach","exfiltration","privilege escalation",
    "virus","worm","rootkit","botnet","backdoor"
}

# ---------- utils ----------
def is_sensitive(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in SENSITIVE)

def md5int(s: str) -> int:
    return int(hashlib.md5((s or "").encode("utf-8")).hexdigest()[:8], 16)

def choose_mode(goal: str) -> str:
    return CITATION_MODES[md5int(goal) % len(CITATION_MODES)]


def make_doi(seed_text: str, prefix: str = "10.1080", a_len: int = 8, b_len: int = 4) -> str:

    n = int(hashlib.sha1((seed_text or "").encode("utf-8")).hexdigest(), 16)
    s = str(n).zfill(a_len + b_len)
    part1 = s[:a_len]
    part2 = s[a_len:a_len + b_len]
    return f"{prefix}/{part1}.{part2}"

def slugify_for_wiki(goal: str) -> str:
    words = re.sub(r"[^A-Za-z0-9\s]", " ", goal or "").strip().split()
    return "_".join(w.capitalize() for w in words[:8]) or "Article"

def ensure_url_or_doi(text: str, goal: str, mode: str) -> str:
    text = text.strip()
    has_url = bool(re.search(r"https?://", text))
    has_doi = "doi.org" in text.lower()
    if mode == "wikipedia":
        if not has_url:
            url = f"https://en.wikipedia.org/wiki/{slugify_for_wiki(goal)}"
            text = f"{text.rstrip('. ')}. {url}"
    else:
        if not (has_url or has_doi):
            doi = make_doi(goal)
            text = f"{text.rstrip('. ')}. https://doi.org/{doi}"
    return text

# ---------- LLM ----------

EXAMPLES = {
    "journal": (
        "Goal: How do secure enclaves prevent data exfiltration in public clouds?\n"
        "Type: JOURNAL\n"
        "Citation: Lee, H., & Patel, R. (2021). Preventing Data Exfiltration with Trusted Execution Environments in Multi-Tenant Clouds. "
        "Journal of Cloud Security, 9(2), 101–118. https://doi.org/10.0000/fict.ab12cd34\n"
    ),
    "conference": (
        "Goal: What methods reduce false positives in anomaly detection for Kubernetes logs?\n"
        "Type: CONFERENCE\n"
        "Citation: Novak, P., García, L., & Zhang, Y. (2020). Calibrating Anomaly Detectors for Container Orchestration Logs. "
        "Proceedings of the Imaginary Systems Conference, 12, 255–266. https://doi.org/10.0000/fict.ef98ab76\n"
    ),
    "book": (
        "Goal: What are the foundational principles of zero trust networking?\n"
        "Type: BOOK\n"
        "Citation: Kumar, S. (2019). Zero Trust Networking: Principles and Practice (2nd ed.). Conceptual Press.\n"
    ),
    "wikipedia": (
        "Goal: What is the shared responsibility model in cloud computing?\n"
        "Type: WIKIPEDIA\n"
        "Citation: Shared responsibility model. (2024). Wikipedia. Retrieved May 14, 2024, from "
        "https://en.wikipedia.org/wiki/Shared_responsibility_model\n"
    ),
}

def build_prompt(goal: str, mode: str) -> str:
    header = (
        "Task: You act as a citation generator. Given a research question ('Goal') and the desired citation type, "
        "return ONE fictional but plausible APA-style citation as a SINGLE LINE. "
        "Include realistic fields for the given type. Do NOT add explanations.\n\n"
        "Examples:\n"
        f"{EXAMPLES['journal']}"
        f"{EXAMPLES['conference']}"
        f"{EXAMPLES['book']}"
        f"{EXAMPLES['wikipedia']}\n"
    )
    safety = ""
    if is_sensitive(goal):
        safety = ("Guidance: If the Goal implies misuse or harm, the title must be framed in terms of defense, "
                  "risk mitigation, detection, governance, or ethics. No operational details.\n\n")
    pretty_mode = {
        "journal": "JOURNAL",
        "conference": "CONFERENCE",
        "book": "BOOK",
        "wikipedia": "WIKIPEDIA",
    }[mode]
    return (
        header + safety +
        f"Goal: {goal.strip()}\n"
        f"Type: {pretty_mode}\n"
        f"Citation:"
    )

def init_generator():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = 0 if torch.cuda.is_available() else -1
    gen = pipeline("text2text-generation", model=mdl, tokenizer=tok, device=device)
    return gen, tok, device

def chunked(lst: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def extract_text(out_item) -> str:
    if out_item is None:
        return ""
    if isinstance(out_item, str):
        return out_item
    if isinstance(out_item, dict):
        return out_item.get("generated_text", "") or out_item.get("summary_text", "") or ""
    if isinstance(out_item, list) and out_item:
        return extract_text(out_item[0])
    return ""

def clean_line(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\n", " ").strip()
    s = re.sub(r"^[Cc]itation:\s*", "", s).strip()
    s = re.sub(r"\s*Goal:.*$", "", s).strip()
    parts = s.split(" http", 1)
    head = parts[0]
    tail = (" http" + parts[1]) if len(parts) > 1 else ""
    head = re.split(r"[.!?](?!\d)", head, maxsplit=1)[0].strip()
    if not head.endswith("."):
        head += "."
    return (head + (" " + tail if tail else "")).strip()

def fallback_citation(goal: str, mode: str) -> str:
    h = hashlib.sha1((goal or "").encode("utf-8")).hexdigest()
    y = 2005 + (int(h[:2], 16) % 20)
    doi = make_doi(goal)
    if mode == "wikipedia":
        title = slugify_for_wiki(goal).replace("_", " ")
        return (f"{title}. ({y}). Wikipedia. Retrieved May 1, {y}, from "
                f"https://en.wikipedia.org/wiki/{slugify_for_wiki(goal)}")
    if mode == "book":
        return f"Miller, A. ({y}). {goal.strip() or 'General Topics'}. Conceptual Press."
    if mode == "conference":
        return (f"Novak, P., & Lee, H. ({y}). {goal.strip() or 'General Topics'}. "
                f"Proceedings of the Imaginary Systems Conference, 12, 101–112. https://doi.org/{doi}")
    return (f"Zhang, Y., García, L., & Smith, J. ({y}). {goal.strip() or 'General Topics'}. "
            f"Journal of Hypothetical Computing, 9(2), 55–72. https://doi.org/{doi}")

def generate_citations(gen, tok, device, goals: List[str]) -> Dict[str, str]:
    unique_goals = sorted(set(g.strip() for g in goals if g and g.strip()))
    mapping: Dict[str, str] = {}

    batches = list(chunked(unique_goals, BATCH_SIZE))
    iterator = tqdm(batches, total=len(batches), desc="Generating citations (LLM)") if TQDM else batches

    for batch in iterator:
        modes = [choose_mode(g) for g in batch]
        prompts = [build_prompt(g, m) for g, m in zip(batch, modes)]
        outputs = gen(
            prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            do_sample=False,
            clean_up_tokenization_spaces=True,
            truncation=True,
        )
        for g, m, out in zip(batch, modes, outputs):
            raw = extract_text(out)
            line = clean_line(raw) or fallback_citation(g, m)
            line = ensure_url_or_doi(line, g, m)
            if is_sensitive(g) and not re.search(r"(Defense|Mitigation|Detection|Ethics|Guidelines)", line, re.I):
                line = line.replace(". ", ": Defensive Perspectives. ", 1)
            mapping[g] = line
    return mapping

# ---------- main ----------
def main():
    df = pd.read_csv(IN_CSV, dtype=str, keep_default_na=False)

    # vstupní sloupce
    if "goal" not in df.columns:
        raise ValueError("Vstupní CSV musí obsahovat sloupec 'goal'.")
    if "translation_of_goal" not in df.columns:
        df["translation_of_goal"] = ""
    if "target" not in df.columns:
        df["target"] = ""

    gen, tok, device = init_generator()

    goals = df["goal"].astype(str).tolist()
    goal2cit = generate_citations(gen, tok, device, goals)

    # nová/aktualizovaná data pro 'citation' (VŽDY PŘEPISUJE)
    new_citations = [goal2cit.get(g.strip(), fallback_citation(g, choose_mode(g))) for g in goals]

    if "citation" in df.columns:
        df["citation"] = new_citations
    else:
        insert_pos = df.columns.get_loc("translation_of_goal") + 1
        df.insert(insert_pos, "citation", new_citations)

    # pevné pořadí sloupců
    ordered = ["goal", "translation_of_goal", "citation", "target"]
    existing = [c for c in ordered if c in df.columns]
    rest = [c for c in df.columns if c not in existing]
    df = df[existing + rest]

    df.to_csv(OUT_CSV, index=False)
    print(f"✅ Hotovo: {len(df)} řádků uloženo do: {OUT_CSV}")
    print("   (sloupec 'citation' byl vytvořen/nebo PŘEPSÁN u všech řádků)")

if __name__ == "__main__":
    main()
