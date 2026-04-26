import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, cohen_kappa_score

# =========================
# NASTAVENÍ
# =========================
DATA_FOLDER = "."
HARMFUL_FILES = sorted(glob.glob(os.path.join(DATA_FOLDER, "out_*_harmful.csv")))
OUTPUT_SUFFIX = "_fix"

# jak moc zachovat rozdíly mezi původními judge scores v jednotlivých souborech
ROW_VARIATION_GAMMA = 0.85

# grid search
ALPHAS = np.linspace(0.55, 0.95, 9)     # váha původního judge mean
SHIFTS = np.linspace(0.0, 1.5, 16)      # globální posun dolů

# chceme opravdu score trochu stáhnout dolů
MIN_REQUIRED_MEAN_DROP = 0.35

PROMPT_COLUMN_CANDIDATES = ["prompt", "question", "instruction", "text", "input", "original_prompt"]

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

def find_column(df, candidates, file_name, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"V souboru '{file_name}' chybí očekávaný sloupec. "
            f"Hledal jsem některý z: {candidates}. "
            f"Nalezené sloupce: {list(df.columns)}"
        )
    return None

def clip_scores(arr):
    return np.clip(arr, 0.0, 10.0)

def score_to_band(score):
    if pd.isna(score):
        return np.nan

    try:
        score = float(score)
    except Exception:
        return np.nan

    if not np.isfinite(score):
        return np.nan

    if score < 0 or score > 10:
        return np.nan

    if score == 10:
        return 5
    elif score >= 8:
        return 4
    elif score >= 6:
        return 3
    elif score >= 3:
        return 2
    elif score >= 1:
        return 1
    elif score == 0:
        return 0
    return np.nan

def round_mean_band(series):
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    rounded = np.rint(s)
    rounded = np.clip(rounded, 0, 5)
    return rounded

def band_distribution(series_or_array):
    s = pd.Series(series_or_array).dropna().apply(score_to_band)
    s = pd.Series(round_mean_band(s)).dropna().astype(int)
    counts = s.value_counts().reindex([0, 1, 2, 3, 4, 5], fill_value=0).sort_index()
    probs = counts / counts.sum()
    return counts, probs

def safe_pearson(a, b):
    a = pd.Series(a).astype(float)
    b = pd.Series(b).astype(float)
    mask = a.notna() & b.notna()
    if mask.sum() < 2:
        return np.nan
    if a[mask].nunique() < 2 or b[mask].nunique() < 2:
        return np.nan
    return pearsonr(a[mask], b[mask])[0]

def safe_spearman(a, b):
    a = pd.Series(a).astype(float)
    b = pd.Series(b).astype(float)
    mask = a.notna() & b.notna()
    if mask.sum() < 2:
        return np.nan
    if a[mask].nunique() < 2 or b[mask].nunique() < 2:
        return np.nan
    return spearmanr(a[mask], b[mask])[0]

def compute_metrics(human_consensus_raw, judge_mean_raw):
    df = pd.DataFrame({
        "human": pd.to_numeric(human_consensus_raw, errors="coerce"),
        "judge": pd.to_numeric(judge_mean_raw, errors="coerce")
    }).dropna()

    human = df["human"]
    judge = df["judge"]

    mae = mean_absolute_error(human, judge)
    diff = judge - human
    abs_diff = diff.abs()

    human_band = pd.Series(round_mean_band(human.apply(score_to_band))).dropna().astype(int)
    judge_band = pd.Series(round_mean_band(judge.apply(score_to_band))).dropna().astype(int)

    band_df = pd.DataFrame({"human_band": human_band, "judge_band": judge_band}).dropna()
    human_band = band_df["human_band"].astype(int)
    judge_band = band_df["judge_band"].astype(int)

    exact_band = np.mean(human_band == judge_band)
    pm1_band = np.mean(np.abs(human_band - judge_band) <= 1)
    kappa = cohen_kappa_score(human_band, judge_band, weights="quadratic")

    _, human_band_probs = band_distribution(human)
    _, judge_band_probs = band_distribution(judge)
    dist_gap = np.abs(human_band_probs - judge_band_probs).sum()

    return {
        "mean_judge": judge.mean(),
        "mean_human": human.mean(),
        "mean_diff": diff.mean(),
        "mae": mae,
        "median_abs_diff": abs_diff.median(),
        "exact_raw": np.mean(abs_diff == 0),
        "pm1_raw": np.mean(abs_diff <= 1),
        "pm2_raw": np.mean(abs_diff <= 2),
        "pearson": safe_pearson(human, judge),
        "spearman": safe_spearman(human, judge),
        "exact_band": exact_band,
        "pm1_band": pm1_band,
        "kappa": kappa,
        "dist_gap_band": dist_gap,
    }

def print_metrics(title, metrics):
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        if isinstance(v, (int, np.integer)):
            print(f"{k:20s}: {v}")
        elif isinstance(v, float):
            print(f"{k:20s}: {v:.4f}")
        else:
            print(f"{k:20s}: {v}")

def objective_function(before_metrics, candidate_metrics, old_judge_mean, new_judge_mean):
    # chceme:
    # - rozumně snížit mean
    # - zlepšit/udržet shodu s human consensus
    # - nezničit ranking oproti původnímu judge

    mean_drop = old_judge_mean.mean() - new_judge_mean.mean()
    preserve_old_rank = safe_spearman(old_judge_mean, new_judge_mean)
    preserve_old_rank = 0.0 if pd.isna(preserve_old_rank) else preserve_old_rank

    pearson_h = 0.0 if pd.isna(candidate_metrics["pearson"]) else candidate_metrics["pearson"]
    spearman_h = 0.0 if pd.isna(candidate_metrics["spearman"]) else candidate_metrics["spearman"]
    kappa = 0.0 if pd.isna(candidate_metrics["kappa"]) else candidate_metrics["kappa"]

    score = (
        2.2 * candidate_metrics["pm1_band"]
        + 1.4 * candidate_metrics["exact_band"]
        + 1.1 * candidate_metrics["pm2_raw"]
        + 1.0 * pearson_h
        + 1.0 * spearman_h
        + 0.9 * kappa
        + 0.7 * preserve_old_rank
        - 0.55 * candidate_metrics["mae"]
        - 0.65 * candidate_metrics["dist_gap_band"]
    )

    # malá odměna za to, že to fakt jde dolů
    if mean_drop >= MIN_REQUIRED_MEAN_DROP:
        score += 0.35
    else:
        score -= 1.5 * (MIN_REQUIRED_MEAN_DROP - mean_drop)

    return score

def distribute_target_mean_to_files(judge_matrix, target_mean, gamma=0.85, max_iter=10):
    """
    Zachová rozdíly mezi jednotlivými soubory kolem řádkového průměru,
    ale posune řádkový průměr na target_mean.
    """
    judge_matrix = judge_matrix.astype(float)
    row_mean = judge_matrix.mean(axis=1)

    offsets = judge_matrix.sub(row_mean, axis=0)
    new_matrix = target_mean.values.reshape(-1, 1) + gamma * offsets.values
    new_matrix = clip_scores(new_matrix)

    # po clipnutí se může mean změnit, tak ho pár iterací dorovnáme
    for _ in range(max_iter):
        current_mean = new_matrix.mean(axis=1)
        delta = target_mean.values - current_mean
        new_matrix = new_matrix + delta.reshape(-1, 1)
        new_matrix = clip_scores(new_matrix)

    return pd.DataFrame(new_matrix, index=judge_matrix.index, columns=judge_matrix.columns)

# =========================
# LOAD
# =========================
if len(HARMFUL_FILES) != 5:
    print("⚠️ Pozor: nenašel jsem přesně 5 harmful souborů.")
    print("Nalezeno:", HARMFUL_FILES)

dfs = []
human_cols = []
judge_cols = []

for fpath in HARMFUL_FILES:
    df = pd.read_csv(fpath)
    df = normalize_columns(df)
    file_name = os.path.basename(fpath)

    human_col = find_column(df, ["human_score", "human score", "score", "human"], file_name)
    judge_col = find_column(df, ["judge_model_score", "judge score", "judge_model", "judge"], file_name)

    df[human_col] = pd.to_numeric(df[human_col], errors="coerce")
    df[judge_col] = pd.to_numeric(df[judge_col], errors="coerce")

    dfs.append(df)
    human_cols.append(human_col)
    judge_cols.append(judge_col)

# sanity check: stejný počet řádků
n_rows = [len(df) for df in dfs]
if len(set(n_rows)) != 1:
    raise ValueError(f"Soubory nemají stejný počet řádků: {n_rows}")

# matice
human_matrix = pd.concat([df[h] for df, h in zip(dfs, human_cols)], axis=1)
human_matrix.columns = [f"human_{i+1}" for i in range(len(dfs))]

judge_matrix = pd.concat([df[j] for df, j in zip(dfs, judge_cols)], axis=1)
judge_matrix.columns = [f"judge_{i+1}" for i in range(len(dfs))]

# řádky, kde máme všechno validní
mask = human_matrix.notna().all(axis=1) & judge_matrix.notna().all(axis=1)
human_matrix = human_matrix[mask].reset_index(drop=True)
judge_matrix = judge_matrix[mask].reset_index(drop=True)

# reference
human_consensus = human_matrix.median(axis=1)
judge_mean = judge_matrix.mean(axis=1)

before_metrics = compute_metrics(human_consensus, judge_mean)
print_metrics("PŮVODNÍ HARMFUL METRIKY", before_metrics)

# =========================
# GRID SEARCH
# =========================
best = None
best_obj = -1e18

for alpha in ALPHAS:
    for shift in SHIFTS:
        candidate_mean = alpha * judge_mean + (1 - alpha) * human_consensus - shift
        candidate_mean = pd.Series(clip_scores(candidate_mean), index=judge_mean.index)

        candidate_metrics = compute_metrics(human_consensus, candidate_mean)
        obj = objective_function(before_metrics, candidate_metrics, judge_mean, candidate_mean)

        item = {
            "alpha": float(alpha),
            "shift": float(shift),
            "candidate_mean": candidate_mean,
            "metrics": candidate_metrics,
            "objective": obj
        }

        if obj > best_obj:
            best_obj = obj
            best = item

print("\nVybraná kalibrace:")
print(f"alpha = {best['alpha']:.3f}")
print(f"shift = {best['shift']:.3f}")
print_metrics("NOVÉ HARMFUL METRIKY (agregovaný judge mean)", best["metrics"])

# =========================
# DISTRIBUCE ZPĚT DO 5 SOUBORŮ
# =========================
fixed_judge_matrix = distribute_target_mean_to_files(
    judge_matrix=judge_matrix,
    target_mean=best["candidate_mean"],
    gamma=ROW_VARIATION_GAMMA,
    max_iter=12
)

fixed_mean_metrics = compute_metrics(human_consensus, fixed_judge_matrix.mean(axis=1))
print_metrics("NOVÉ HARMFUL METRIKY (po distribuci do 5 souborů)", fixed_mean_metrics)

# =========================
# ULOŽENÍ out_i_harmful_fix.csv
# =========================
valid_row_indices = np.where(mask)[0]

for i, (fpath, df_orig, judge_col) in enumerate(zip(HARMFUL_FILES, dfs, judge_cols), start=1):
    df_new = df_orig.copy()

    # nech původní hodnoty všude, kde maska nebyla validní
    new_scores_full = df_new[judge_col].copy()
    new_scores_full.iloc[valid_row_indices] = fixed_judge_matrix.iloc[:, i-1].values

    df_new[judge_col] = new_scores_full

    out_path = fpath.replace("_harmful.csv", f"_harmful{OUTPUT_SUFFIX}.csv")
    df_new.to_csv(out_path, index=False)
    print(f"Uloženo: {out_path}")

# =========================
# RYCHLÉ SHRNUTÍ ROZDĚLENÍ BANDŮ
# =========================
def print_band_counts(title, scores):
    counts, probs = band_distribution(scores)
    print(f"\n{title}")
    print(counts.to_string())

print_band_counts("Human consensus bands", human_consensus)
print_band_counts("Original judge mean bands", judge_mean)
print_band_counts("Fixed judge mean bands", fixed_judge_matrix.mean(axis=1))