# -*- coding: utf-8 -*-
"""
Výpočet metrik pro 5 anotátorů ze souborů CSV se sloupcem `human_score`.

Co skript dělá:
1) Deskriptivní statistika:
   - mean, median, mode
   - skewness, kurtosis
   - z-score normalizace
   - histogram + KDE všech anotátorů

2) Inter-Annotator Agreement:
   - Krippendorff's alpha (ordinal)
   - 95% CI pro alpha pomocí bootstrapu
   - Kendall's W
   - pairwise Cohen's kappa (quadratic weighted)
   - heatmapa pairwise kappa
   - percent agreement (exact a ±1)

3) Analýza chyb a konsenzu:
   - Leave-One-Out alpha
   - Bland-Altman plot
   - Hard cases top 20
   - Majority ratio

Předpoklady:
- Máš 5 CSV souborů.
- Každý soubor má sloupec `human_score`.
- Řádky jsou ve stejném pořadí napříč soubory.
  Pokud máš identifikátor řádku (např. id), doporučuju merge podle id.

Instalace:
pip install pandas numpy scipy matplotlib scikit-learn seaborn krippendorff
"""

from pathlib import Path
from collections import Counter
import itertools
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, zscore
from sklearn.metrics import cohen_kappa_score

try:
    import krippendorff
except ImportError:
    raise ImportError("Nainstaluj balíček: pip install krippendorff")


# =========================
# NASTAVENÍ
# =========================

# Sem dej názvy svých 5 CSV souborů:
FILES = [
    "show_1.csv",
    "show_2.csv",
    "show_3.csv",
    "show_4.csv",
    "show_5.csv",
]

SCORE_COL = "human_score"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

BOOTSTRAP_ITER = 2000
RANDOM_SEED = 42


# =========================
# NAČTENÍ DAT
# =========================

def load_scores(files, score_col="human_score"):
    dfs = []
    for i, f in enumerate(files, start=1):
        df = pd.read_csv(f)
        if score_col not in df.columns:
            raise ValueError(f"Soubor {f} neobsahuje sloupec '{score_col}'.")
        s = df[score_col].copy()
        dfs.append(s.rename(f"annotator_{i}"))

    ratings = pd.concat(dfs, axis=1)

    # Vyhodíme řádky, kde někdo nemá hodnocení
    ratings = ratings.dropna().reset_index(drop=True)

    return ratings


ratings = load_scores(FILES, SCORE_COL)
annotators = ratings.columns.tolist()

print("Načtená data:")
print(ratings.head())
print(f"\nPočet společných anotovaných řádků: {len(ratings)}")


# =========================
# 1. DESKRIPTIVNÍ STATISTIKA
# =========================

def safe_mode(series):
    modes = series.mode(dropna=True)
    if len(modes) == 0:
        return np.nan
    if len(modes) == 1:
        return modes.iloc[0]
    return list(modes.values)

desc_rows = []
for col in annotators:
    s = ratings[col].dropna()
    desc_rows.append({
        "annotator": col,
        "count": len(s),
        "mean": s.mean(),
        "median": s.median(),
        "mode": safe_mode(s),
        "std": s.std(ddof=1),
        "min": s.min(),
        "max": s.max(),
        "skewness": skew(s, bias=False),
        "kurtosis": kurtosis(s, fisher=True, bias=False),  # excess kurtosis
    })

desc_df = pd.DataFrame(desc_rows)
desc_df.to_csv(OUTPUT_DIR / "descriptive_statistics.csv", index=False)

print("\n=== Deskriptivní statistika ===")
print(desc_df)

# Z-score normalizace po anotátorech
zscore_df = ratings.copy()
for col in annotators:
    # pokud by měl anotátor konstantní hodnoty, zscore by dělal NaN
    if ratings[col].std(ddof=0) == 0:
        zscore_df[col] = 0.0
    else:
        zscore_df[col] = zscore(ratings[col], ddof=0)

zscore_df.to_csv(OUTPUT_DIR / "zscore_normalized_scores.csv", index=False)

# Histogram + KDE-like vizualizace (přes hustotu histogramu)
plt.figure(figsize=(10, 6))
bins = sorted(pd.unique(ratings.values.ravel()))
if len(bins) < 2:
    bins = 10

for col in annotators:
    ratings[col].plot(kind="hist", density=True, alpha=0.25, bins=bins, label=col)

# jednoduchý line KDE přes pandas plot density
for col in annotators:
    try:
        ratings[col].plot(kind="density", linewidth=2, label=f"{col} KDE")
    except Exception:
        pass

plt.title("Distribuce skóre všech anotátorů")
plt.xlabel("human_score")
plt.ylabel("Hustota")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "histogram_kde_annotators.png", dpi=200)
plt.close()


# =========================
# 2. INTER-ANNOTATOR AGREEMENT
# =========================

# 2.1 Krippendorff's alpha (ordinal)
def kripp_alpha_ordinal(df):
    # krippendorff očekává tvar [annotators, items]
    data = df.to_numpy().T
    return krippendorff.alpha(reliability_data=data, level_of_measurement="ordinal")

alpha = kripp_alpha_ordinal(ratings)
print(f"\nKrippendorff's alpha (ordinal): {alpha:.4f}")

# 2.2 Bootstrap 95% CI pro alpha
def bootstrap_alpha_ci(df, n_iter=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(df)
    vals = []

    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        sample = df.iloc[idx].reset_index(drop=True)
        try:
            a = kripp_alpha_ordinal(sample)
            if not np.isnan(a):
                vals.append(a)
        except Exception:
            pass

    vals = np.array(vals)
    lower = np.percentile(vals, 2.5)
    upper = np.percentile(vals, 97.5)
    return vals.mean(), lower, upper, vals

alpha_boot_mean, alpha_ci_low, alpha_ci_high, alpha_boot_vals = bootstrap_alpha_ci(
    ratings, n_iter=BOOTSTRAP_ITER, seed=RANDOM_SEED
)

print(f"95% CI pro alpha: [{alpha_ci_low:.4f}, {alpha_ci_high:.4f}]")

pd.DataFrame({"alpha_bootstrap": alpha_boot_vals}).to_csv(
    OUTPUT_DIR / "krippendorff_alpha_bootstrap_distribution.csv", index=False
)

# 2.3 Kendall's W
def kendalls_w(df):
    """
    df: rows = items, cols = raters
    Kendall's W pro shodu v pořadí.
    """
    X = df.to_numpy()
    n, m = X.shape  # n items, m raters

    # Ranks within each rater column
    rank_df = pd.DataFrame(X).rank(axis=0, method="average")
    R = rank_df.sum(axis=1)
    R_bar = R.mean()

    S = ((R - R_bar) ** 2).sum()
    W = 12 * S / (m**2 * (n**3 - n))
    return W

W = kendalls_w(ratings)
print(f"Kendall's W: {W:.4f}")

# 2.4 Pairwise Cohen's Kappa (quadratic weighted)
kappa_matrix = pd.DataFrame(index=annotators, columns=annotators, dtype=float)

for a, b in itertools.product(annotators, annotators):
    if a == b:
        kappa_matrix.loc[a, b] = 1.0
    else:
        kappa_matrix.loc[a, b] = cohen_kappa_score(
            ratings[a], ratings[b], weights="quadratic"
        )

kappa_matrix.to_csv(OUTPUT_DIR / "pairwise_weighted_kappa_matrix.csv")

print("\n=== Pairwise weighted Cohen's Kappa ===")
print(kappa_matrix)

# heatmapa bez seabornu
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(kappa_matrix.values.astype(float))

ax.set_xticks(range(len(annotators)))
ax.set_yticks(range(len(annotators)))
ax.set_xticklabels(annotators, rotation=45, ha="right")
ax.set_yticklabels(annotators)

for i in range(len(annotators)):
    for j in range(len(annotators)):
        ax.text(j, i, f"{kappa_matrix.iloc[i, j]:.2f}",
                ha="center", va="center")

plt.title("Pairwise quadratic weighted Cohen's Kappa")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pairwise_kappa_heatmap.png", dpi=200)
plt.close()

# 2.5 Percent agreement
def percent_agreement(df, tolerance=0):
    X = df.to_numpy()
    n_items, n_raters = X.shape
    total_pairs = math.comb(n_raters, 2) * n_items

    agree = 0
    for row in X:
        for i, j in itertools.combinations(range(n_raters), 2):
            if abs(row[i] - row[j]) <= tolerance:
                agree += 1

    return agree / total_pairs

percent_exact = percent_agreement(ratings, tolerance=0)
percent_pm1 = percent_agreement(ratings, tolerance=1)
percent_pm2 = percent_agreement(ratings, tolerance=2)

agreement_df = pd.DataFrame({
    "metric": ["percent_agreement_exact", "percent_agreement_pm1", "percent_agreement_pm2"],
    "value": [percent_exact, percent_pm1, percent_pm2]
})
agreement_df.to_csv(OUTPUT_DIR / "percent_agreement.csv", index=False)

print(f"\nPercent agreement exact: {percent_exact:.4f}")
print(f"Percent agreement ±1:   {percent_pm1:.4f}")
print(f"Percent agreement ±2:   {percent_pm2:.4f}")


# =========================
# 3. ANALÝZA CHYB A KONSENZU
# =========================

# 3.1 Leave-One-Out alpha
loo_rows = []
for col in annotators:
    subset = ratings.drop(columns=[col])
    loo_alpha = kripp_alpha_ordinal(subset)
    loo_rows.append({
        "left_out": col,
        "alpha_without_annotator": loo_alpha,
        "delta_vs_full": loo_alpha - alpha
    })

loo_df = pd.DataFrame(loo_rows).sort_values("alpha_without_annotator", ascending=False)
loo_df.to_csv(OUTPUT_DIR / "leave_one_out_alpha.csv", index=False)

print("\n=== Leave-One-Out alpha ===")
print(loo_df)

# 3.2 Hard cases + majority ratio
def majority_ratio(row):
    counts = Counter(row)
    return max(counts.values()) / len(row)

row_stats = pd.DataFrame({
    "mean_score": ratings.mean(axis=1),
    "median_score": ratings.median(axis=1),
    "std_score": ratings.std(axis=1, ddof=1),
    "range_score": ratings.max(axis=1) - ratings.min(axis=1),
    "majority_ratio": ratings.apply(lambda r: majority_ratio(r.values), axis=1),
})

hard_cases_std = row_stats.sort_values("std_score", ascending=False).head(20).copy()
hard_cases_range = row_stats.sort_values("range_score", ascending=False).head(20).copy()

hard_cases_std = pd.concat([ratings.loc[hard_cases_std.index], hard_cases_std], axis=1)
hard_cases_range = pd.concat([ratings.loc[hard_cases_range.index], hard_cases_range], axis=1)

hard_cases_std.to_csv(OUTPUT_DIR / "hard_cases_top20_by_std.csv", index=True)
hard_cases_range.to_csv(OUTPUT_DIR / "hard_cases_top20_by_range.csv", index=True)

print("\n=== Top 20 hard cases podle std ===")
print(hard_cases_std.head())

print("\n=== Top 20 hard cases podle range ===")
print(hard_cases_range.head())

# 3.3 Bland-Altman plot
# Uděláme ho jako: každý anotátor proti průměru ostatních anotátorů
bland_rows = []
for col in annotators:
    others = ratings.drop(columns=[col]).mean(axis=1)
    annot = ratings[col]
    mean_pair = (annot + others) / 2
    diff_pair = annot - others

    tmp = pd.DataFrame({
        "annotator": col,
        "mean_of_annotator_and_others": mean_pair,
        "difference_annotator_minus_others": diff_pair
    })
    bland_rows.append(tmp)

bland_df = pd.concat(bland_rows, ignore_index=True)
bland_df.to_csv(OUTPUT_DIR / "bland_altman_data.csv", index=False)

plt.figure(figsize=(10, 6))
for col in annotators:
    part = bland_df[bland_df["annotator"] == col]
    plt.scatter(
        part["mean_of_annotator_and_others"],
        part["difference_annotator_minus_others"],
        alpha=0.5,
        label=col
    )

mean_diff = bland_df["difference_annotator_minus_others"].mean()
sd_diff = bland_df["difference_annotator_minus_others"].std(ddof=1)
loa_upper = mean_diff + 1.96 * sd_diff
loa_lower = mean_diff - 1.96 * sd_diff

plt.axhline(mean_diff, linestyle="--", label=f"Mean diff = {mean_diff:.2f}")
plt.axhline(loa_upper, linestyle=":")
plt.axhline(loa_lower, linestyle=":")

plt.title("Bland-Altman plot: anotátor vs průměr ostatních")
plt.xlabel("Průměr (anotátor a průměr ostatních)")
plt.ylabel("Rozdíl (anotátor - průměr ostatních)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bland_altman_plot.png", dpi=200)
plt.close()


# =========================
# SOUHRN
# =========================

summary = pd.DataFrame({
    "metric": [
        "krippendorff_alpha_ordinal",
        "krippendorff_alpha_bootstrap_mean",
        "krippendorff_alpha_ci_low_95",
        "krippendorff_alpha_ci_high_95",
        "kendalls_w",
        "percent_agreement_exact",
        "percent_agreement_pm1",
        "percent_agreement_pm2"
    ],
    "value": [
        alpha,
        alpha_boot_mean,
        alpha_ci_low,
        alpha_ci_high,
        W,
        percent_exact,
        percent_pm1,
        percent_pm2
    ],
})

print(f"\nHotovo. Výstupy najdeš ve složce: {OUTPUT_DIR.resolve()}")