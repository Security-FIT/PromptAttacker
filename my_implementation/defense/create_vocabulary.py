from transformers import AutoTokenizer
import os
import json
import re
import string

# --- staticky zadaný název modelu a lokální cesta ---
MODEL_NAME = "internlm2_5-7b-chat-1m"  # jen pro jméno výstupního souboru
MODEL_DIR = f"/storage/brno2/home/xkaska01/master/my_implementation/models/{MODEL_NAME}"

# --- výstupní soubor ---
OUTPUT_FILE = f"/storage/brno2/home/xkaska01/master/my_implementation/defense/models_vocabularies/{MODEL_NAME}_vocab.txt"

print(f"Načítám tokenizer z lokální cesty: {MODEL_DIR}")

if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(f"Adresář neexistuje: {MODEL_DIR}")

# ↳ Llama tokenizer často potřebuje sentencepiece; local_files_only=True zajistí, že to nesáhne na HF.
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    use_fast=False  # u Llamy bývá spolehlivé 'slow' (SentencePiece)
)

# --- Získání mapy token -> id ---
vocab = tokenizer.get_vocab()

# --- Filtrace netisknutelných / hexových tokenů ---------------------------------------

HEX_PATTERN = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")

def is_printable_token(tok: str) -> bool:
    """Vrací True, pokud je token bezpečně tisknutelný a není pseudohex zápis."""
    # prázdné nebo whitespace = zahodit
    if not tok or tok.isspace():
        return False
    # pseudohex typu <0xF9>
    if HEX_PATTERN.match(tok):
        return False
    # obsahuje-li netisknutelné znaky, také zahodit
    for ch in tok:
        if ord(ch) < 32 or ord(ch) == 127:
            return False
    return True

clean_vocab = {t: i for t, i in vocab.items() if is_printable_token(t)}

removed = len(vocab) - len(clean_vocab)
print(f"[INFO] Odstraněno {removed} netisknutelných tokenů ({len(clean_vocab)} ponecháno).")

# --- Ukládání ------------------------------------------------------------

def safe_token(t: str) -> str:
    """Vrátí JSON-escaped token pro bezpečné uložení."""
    return json.dumps(t, ensure_ascii=False)

# Seřadíme podle id (hodnoty)
items = sorted(clean_vocab.items(), key=lambda kv: kv[1])

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for token, idx in items:
        f.write(f"{idx}\t{safe_token(token)}\n")

print(f"✅ Slovník uložen do: {OUTPUT_FILE}")
