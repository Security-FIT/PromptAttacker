import argparse
import os
import json
import re

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

def safe_token(t: str) -> str:
    """Vrátí JSON-escaped token pro bezpečné uložení."""
    return json.dumps(t, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Create a clean tokenizer vocabulary file for defense training.")
    parser.add_argument(
        "--model-name",
        default=os.getenv("DEFENSE_VOCAB_MODEL_NAME", "internlm2.5:latest"),
        help="Model name used mainly for the default output filename.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Local tokenizer/model directory. Defaults to prompt_attacker/models/<model-name>.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output vocab .txt file. Defaults to defense/models_vocabularies/<model-name>_vocab.txt.",
    )
    args = parser.parse_args()

    model_dir = args.model_dir or os.getenv(
        "DEFENSE_VOCAB_MODEL_DIR",
        f"/storage/brno2/home/xkaska01/master/prompt_attacker/models/{args.model_name}",
    )
    output_file = args.output_file or os.getenv(
        "DEFENSE_VOCAB_OUTPUT_FILE",
        f"/storage/brno2/home/xkaska01/master/prompt_attacker/defense/models_vocabularies/{args.model_name}_vocab.txt",
    )

    print(f"Načítám tokenizer z lokální cesty: {model_dir}")

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Adresář neexistuje: {model_dir}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        use_fast=False,
    )

    vocab = tokenizer.get_vocab()
    clean_vocab = {t: i for t, i in vocab.items() if is_printable_token(t)}

    removed = len(vocab) - len(clean_vocab)
    print(f"[INFO] Odstraněno {removed} netisknutelných tokenů ({len(clean_vocab)} ponecháno).")

    items = sorted(clean_vocab.items(), key=lambda kv: kv[1])
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for token, idx in items:
            f.write(f"{idx}\t{safe_token(token)}\n")

    print(f"✅ Slovník uložen do: {output_file}")


if __name__ == "__main__":
    main()
