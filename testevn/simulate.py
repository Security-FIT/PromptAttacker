import os
import glob
import shutil
import numpy as np
import pandas as pd

# =========================
# NASTAVENÍ
# =========================
DATA_FOLDER = "."
FILE_PATTERN = os.path.join(DATA_FOLDER, "out_*_harmful_fix_updated_simulated.csv")

OVERWRITE = True          # False -> vytvoří *_simulated.csv, True -> přepíše původní soubory
CREATE_BACKUP = False       # používá se jen při OVERWRITE=True
SEED = 42

HUMAN_SCORE_CANDIDATES = ["human_score", "human score", "score", "human"]

rng = np.random.default_rng(SEED)

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

def make_output_path(path, overwrite=False):
    if overwrite:
        return path
    return path.replace(".csv", "_simulated.csv")

def maybe_backup(path):
    if CREATE_BACKUP and OVERWRITE:
        backup_path = path.replace(".csv", "_backup.csv")
        shutil.copy(path, backup_path)
        print(f"Záloha vytvořena: {backup_path}")

def choose_fraction_indices(mask, fraction):
    eligible = np.where(mask)[0]
    if len(eligible) == 0:
        return np.array([], dtype=int)

    n = int(round(len(eligible) * fraction))
    n = max(0, min(n, len(eligible)))

    if n == 0:
        return np.array([], dtype=int)

    return rng.choice(eligible, size=n, replace=False)

def choose_uniform_fraction_indices(mask, min_frac, max_frac):
    eligible = np.where(mask)[0]
    if len(eligible) == 0:
        return np.array([], dtype=int)

    frac = rng.uniform(min_frac, max_frac)
    n = int(round(len(eligible) * frac))
    n = max(1, min(n, len(eligible)))

    return rng.choice(eligible, size=n, replace=False)

def detect_annotator_role(file_name):
    # očekává např. out_2_harmful.csv nebo out_3_harmful_fix_updated.csv
    base = os.path.basename(file_name)
    parts = base.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None

# =========================
# SYSTEMATICKÝ BIAS
# =========================
def transform_annotator_2_strict(scores):
    # Strict: bez změny
    scores = scores.copy()
    return scores, {
        "changed_rows": 0,
        "note": "Strict: bez systematické změny"
    }

def transform_annotator_3_conservative(scores):
    # Conservative:
    # - hodnoty 10 -> 9 v 10 % případů
    # - hodnoty 0  -> 1 v 10 % případů
    scores = scores.copy()
    changed = 0

    idx_10 = choose_fraction_indices(scores.notna() & (scores == 10), 0.10)
    for idx in idx_10:
        scores.iloc[idx] = 9
        changed += 1

    idx_0 = choose_fraction_indices(scores.notna() & (scores == 0), 0.10)
    for idx in idx_0:
        scores.iloc[idx] = 1
        changed += 1

    return scores, {
        "changed_rows": changed,
        "note": "Conservative: 10->9 a 0->1 v 10 % případů"
    }

def transform_annotator_4_alarmist(scores):
    # Alarmist:
    # - jakmile je skóre >= 7, posune na 10
    scores = scores.copy()
    mask = scores.notna() & (scores >= 7) & (scores != 10)
    changed = int(mask.sum())
    scores.loc[mask] = 10

    return scores, {
        "changed_rows": changed,
        "note": "Alarmist: všechno >=7 posunuto na 10"
    }

def transform_annotator_5_nuanced(scores):
    # Nuanced:
    # - u 7,8,9 v 20 % případů udělá +/-1
    scores = scores.copy()
    mask = scores.notna() & scores.isin([7, 8, 9])
    chosen = choose_fraction_indices(mask, 0.20)

    for idx in chosen:
        delta = int(rng.choice([-1, 1]))
        scores.iloc[idx] = clip_score(scores.iloc[idx] + delta)

    return scores, {
        "changed_rows": len(chosen),
        "note": "Nuanced: u 7/8/9 v 20 % případů změna o +/-1"
    }

def apply_role_transform(role, scores):
    if role == 2:
        return transform_annotator_2_strict(scores)
    elif role == 3:
        return transform_annotator_3_conservative(scores)
    elif role == 4:
        return transform_annotator_4_alarmist(scores)
    elif role == 5:
        return transform_annotator_5_nuanced(scores)
    else:
        return scores.copy(), {
            "changed_rows": 0,
            "note": "Nedefinovaná role"
        }

# =========================
# KOTEVNÍ EFEKT / PREFEROVANÁ ČÍSLA
# =========================
def apply_anchoring_bias(scores):
    """
    Lehce upraví 8 a 9 tak, aby 9 bylo o něco méně časté a 8 o něco častější.
    Implementace:
    - 9 -> 8 v 15 % případů
    - 7 -> 8 v 10 % případů
    """
    scores = scores.copy()
    changed = 0

    idx_9 = choose_fraction_indices(scores.notna() & (scores == 9), 0.15)
    for idx in idx_9:
        scores.iloc[idx] = 8
        changed += 1

    idx_7 = choose_fraction_indices(scores.notna() & (scores == 7), 0.10)
    for idx in idx_7:
        scores.iloc[idx] = 8
        changed += 1

    return scores, {
        "changed_rows": changed,
        "note": "Anchoring bias: 9->8 v 15 %, 7->8 v 10 %"
    }

# =========================
# NÁHODNÝ ŠUM / LIDSKÁ CHYBA
# =========================
def apply_random_noise(scores, min_frac=0.03, max_frac=0.05):
    """
    U náhodně vybraných 3–5 % řádků změní skóre o náhodné ±2 body.
    """
    scores = scores.copy()
    chosen = choose_uniform_fraction_indices(scores.notna(), min_frac, max_frac)

    for idx in chosen:
        delta = int(rng.choice([-2, 2]))
        scores.iloc[idx] = clip_score(scores.iloc[idx] + delta)

    return scores, {
        "changed_rows": len(chosen),
        "note": "Random noise: 3–5 % řádků změněno o ±2"
    }

# =========================
# HLAVNÍ LOGIKA
# =========================
def process_file(path):
    print("\n" + "=" * 90)
    print(f"Zpracovávám: {path}")

    df = pd.read_csv(path)
    df = normalize_columns(df)

    score_col = find_first_existing_column(df, HUMAN_SCORE_CANDIDATES, required=True)
    original_scores = pd.to_numeric(df[score_col], errors="coerce").copy()

    role = detect_annotator_role(path)
    if role not in [2, 3, 4, 5]:
        print(f"Přeskakuji {path} — nemá roli 2/3/4/5.")
        return

    # 1) systematický bias dle role
    role_scores, role_info = apply_role_transform(role, original_scores)

    # 2) anchoring bias
    anchored_scores, anchor_info = apply_anchoring_bias(role_scores)

    # 3) random noise
    final_scores, noise_info = apply_random_noise(anchored_scores, min_frac=0.03, max_frac=0.05)

    final_scores = final_scores.apply(clip_score)

    changed_vs_original = int(
        (original_scores.fillna(-9999) != final_scores.fillna(-9999)).sum()
    )

    df[score_col] = final_scores

    if OVERWRITE:
        maybe_backup(path)

    output_path = make_output_path(path, overwrite=OVERWRITE)
    df.to_csv(output_path, index=False)

    print(f"Role anotátora: {role}")
    print(f"Systematický bias: {role_info['note']}")
    print(f"Změněno systematickým biasem: {role_info['changed_rows']}")
    print(f"Anchoring bias: {anchor_info['note']}")
    print(f"Změněno anchoring biasem: {anchor_info['changed_rows']}")
    print(f"Random noise: {noise_info['note']}")
    print(f"Změněno random noise: {noise_info['changed_rows']}")
    print(f"Celkem změněných řádků oproti originálu: {changed_vs_original}")
    print(f"Uloženo do: {output_path}")

def main():
    files = sorted(glob.glob(FILE_PATTERN))

    if not files:
        raise FileNotFoundError(f"Nebyly nalezeny soubory pro pattern: {FILE_PATTERN}")

    print("Nalezené harmful soubory:")
    for f in files:
        print(f" - {os.path.basename(f)}")

    for path in files:
        process_file(path)

    print("\nHotovo.")

if __name__ == "__main__":
    main()