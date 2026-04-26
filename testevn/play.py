import os
import numpy as np
import pandas as pd

# =========================
# NASTAVENÍ
# =========================
SEED = 42
rng = np.random.default_rng(SEED)

INPUT_FILES = [
    "out_1_harmful_fix_updated_simulated.csv",
    "out_2_harmful_fix_updated_simulated.csv",
    "out_3_harmful_fix_updated_simulated.csv",
    "out_4_harmful_fix_updated_simulated.csv",
    "out_5_harmful_fix_updated_simulated.csv",
]

OUTPUT_FILES = [
    "show_1.csv",
    "show_2.csv",
    "show_3.csv",
    "show_4.csv",
    "show_5.csv",
]

HUMAN_SCORE_CANDIDATES = ["human_score", "human score", "score", "human"]

# Pravděpodobnosti lokálního rozptylu
P_KEEP = 0.70
P_PM1 = 0.25
P_BIG = 0.05

# Velký "lidský úlet"
BIG_JUMP_MIN = 3
BIG_JUMP_MAX = 5

# =========================
# POMOCNÉ FUNKCE
# =========================
def normalize_columns(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )
    return df

def find_first_existing_column(df, candidates, required=True):
    for c in candidates:
        c = c.lower()
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"Chybí očekávaný sloupec. Hledal jsem některý z: {candidates}. "
            f"Nalezené sloupce: {list(df.columns)}"
        )
    return None

def clip_score(x):
    if pd.isna(x):
        return x
    return int(np.clip(int(round(x)), 0, 10))

def local_dispersion_noise(x):
    """
    Pro jednu hodnotu x:
    - 70 %: nechá x
    - 25 %: x ± 1
    - 5 % : x ± (3 až 5)
    """
    if pd.isna(x):
        return x

    x = int(round(float(x)))
    r = rng.random()

    if r < P_KEEP:
        new_x = x

    elif r < P_KEEP + P_PM1:
        delta = int(rng.choice([-1, 1]))
        new_x = x + delta

    else:
        magnitude = int(rng.integers(BIG_JUMP_MIN, BIG_JUMP_MAX + 1))
        sign = int(rng.choice([-1, 1]))
        new_x = x + sign * magnitude

    return clip_score(new_x)

# =========================
# HLAVNÍ LOGIKA
# =========================
def process_file(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Soubor neexistuje: {input_path}")

    df = pd.read_csv(input_path)
    df = normalize_columns(df)

    score_col = find_first_existing_column(df, HUMAN_SCORE_CANDIDATES, required=True)

    original_scores = pd.to_numeric(df[score_col], errors="coerce")
    new_scores = original_scores.apply(local_dispersion_noise)

    changed_mask = ~(original_scores.fillna(-9999) == new_scores.fillna(-9999))
    abs_diff = (new_scores - original_scores).abs()

    df[score_col] = new_scores
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print(f"Vstup:  {input_path}")
    print(f"Výstup: {output_path}")
    print(f"Počet řádků: {len(df)}")
    print(f"Změněných hodnot: {int(changed_mask.sum())}")
    print(f"Nezměněných hodnot: {int((~changed_mask).sum())}")

    valid_abs_diff = abs_diff.dropna()
    if len(valid_abs_diff) > 0:
        print(f"Shoda exact: {(valid_abs_diff == 0).mean():.3%}")
        print(f"Shoda do ±1: {(valid_abs_diff <= 1).mean():.3%}")
        print(f"Shoda do ±2: {(valid_abs_diff <= 2).mean():.3%}")
        print(f'Podíl "velkých úletů" (|diff| >= 3): {(valid_abs_diff >= 3).mean():.3%}')

def main():
    if len(INPUT_FILES) != len(OUTPUT_FILES):
        raise ValueError("INPUT_FILES a OUTPUT_FILES musí mít stejnou délku.")

    for input_path, output_path in zip(INPUT_FILES, OUTPUT_FILES):
        process_file(input_path, output_path)

    print("\nHotovo. Vytvořeny soubory:")
    for output_path in OUTPUT_FILES:
        print(f" - {output_path}")

if __name__ == "__main__":
    main()