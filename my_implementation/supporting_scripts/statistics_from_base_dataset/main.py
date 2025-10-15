#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analýza dvou částí datasetu z pevné cesty:
- Prvních 500 řádků -> CySec (červená)
- Dalších 619 řádků -> Alpaca (modrá)

CSV: /storage/brno2/home/xkaska01/master/my_implementation/dataset/base_dataset.csv

Vytváří:
- Koláč: CySec vs Alpaca (červená vs modrá) s dynamickými popisky
- Prstencový koláč: vnitřní (datasety), vnější (kategorie v rámci datasetů) — VŠECHNY názvy mimo
- Koláče kategorií (vylepšené: malé výseče mají procento mimo, žádné překryvy)
- Porovnání kategorií (tabulky + bar)
- Kybercrimes vs obecne_crimes (porovnání + koláče)
- Statistika délek (describe + histogram overlay + boxplot + scatter)
- Report + CSV výstupy

Bez argumentů v příkazové řádce.
"""

import os
import sys
import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from datetime import datetime
import matplotlib as mpl

FS_SM  = 14
FS_MD  = 16
FS_LG  = 18
FS_XL  = 22

mpl.rcParams.update({
    "font.size": FS_MD,          # default pro vše
    "axes.titlesize": FS_XL,
    "axes.labelsize": FS_LG,
    "xtick.labelsize": FS_MD,
    "ytick.labelsize": FS_MD,
    "legend.fontsize": FS_MD,
    "figure.titlesize": FS_XL,
})

# ----------------------------- Konstanty ----------------------------- #
INPUT_CSV = "/storage/brno2/home/xkaska01/master/my_implementation/dataset/base_dataset.csv"
CYSEC_N = 500
ALPACA_N = 619
TOTAL_EXPECTED = CYSEC_N + ALPACA_N

# Které kategorie bereme jako "kybercrimes"
CYBER_CATEGORIES = {
    "malware", "malware attacks", "hacking", "iot attacks", "intrusion techniques",
    "hardware attacks", "evasion techniques", "cryptographic attacks",
    "control system attacks", "cloud attacks", "web application attacks",
    "network attacks"
}

# ----------------------------- Utility ----------------------------- #
def ensure_outputs_dir(path="outputs"):
    os.makedirs(path, exist_ok=True)
    return path

def save_fig(fig, path):
    try:
        fig.tight_layout()
    except Exception:
        plt.subplots_adjust(left=0.06, right=0.88, top=0.92, bottom=0.06)
    fig.savefig(path, dpi=150)
    plt.close(fig)

def normalize_label(label: str) -> str:
    if not isinstance(label, str):
        return ""
    return re.sub(r"\s+", " ", label.strip().lower())

def explode_multilabels(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=str)
    tmp = series.fillna("").astype(str)
    tmp = tmp.str.replace(r"[;|/]", ",", regex=True)
    exploded = tmp.str.split(",").explode().map(normalize_label)
    return exploded[exploded != ""]

def assign_bucket(category: str) -> str:
    if not category:
        return "other"
    return "kybercrimes" if category in CYBER_CATEGORIES else "obecne_crimes"

def safe_len_words(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(text.split())

def safe_len_chars(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(text)

def lighten_color(hex_color, factor=0.5):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"

base_red = "#e41a1c"
base_blue = "#377eb8"

# ----------------------------- Načtení ----------------------------- #
if not os.path.isfile(INPUT_CSV):
    print(f"❌ CSV nenalezeno: {INPUT_CSV}")
    sys.exit(1)

df = pd.read_csv(INPUT_CSV)

required_cols = ["goal", "translation_of_goal", "citation", "target"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"⚠️ Chybí sloupce: {missing}. Pokračuji s dostupnými.")

# Guard na minimální velikost
if len(df) < CYSEC_N + ALPACA_N:
    print(f"⚠️ Pozor: dataset má {len(df)} řádků, očekáváno {TOTAL_EXPECTED}. "
          f"Rozdělím: prvních {CYSEC_N} = CySec, zbytek = Alpaca.")
# Rozdělení
df_cysec = df.iloc[:CYSEC_N].copy()
df_alpaca = df.iloc[CYSEC_N:CYSEC_N+ALPACA_N].copy()  # přesně 619, pokud je k dispozici

# ----------------------------- Inženýring délek ----------------------------- #
for part in (df_cysec, df_alpaca):
    part["goal_len_chars"] = part["goal"].map(safe_len_chars) if "goal" in part else 0
    part["goal_len_words"] = part["goal"].map(safe_len_words) if "goal" in part else 0
    if "translation_of_goal" in part:
        part["trans_len_chars"] = part["translation_of_goal"].map(safe_len_chars)
        part["trans_len_words"] = part["translation_of_goal"].map(safe_len_words)
    else:
        part["translation_of_goal"] = ""
        part["trans_len_chars"] = 0
        part["trans_len_words"] = 0
    part["ratio_trans_to_goal_chars"] = np.where(
        part["goal_len_chars"] > 0,
        part["trans_len_chars"] / part["goal_len_chars"],
        np.nan,
    )
    part["ratio_trans_to_goal_words"] = np.where(
        part["goal_len_words"] > 0,
        part["trans_len_words"] / part["goal_len_words"],
        np.nan,
    )

# ----------------------------- Kategorie ----------------------------- #
def categories_table(df_part: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    labels = explode_multilabels(df_part["target"]) if "target" in df_part else pd.Series(dtype=str)
    cat_counts = labels.value_counts().rename_axis("category").reset_index(name="count")
    if cat_counts.empty:
        return pd.DataFrame(columns=["dataset", "category", "count", "bucket"])
    cat_counts["bucket"] = cat_counts["category"].map(assign_bucket)
    cat_counts["dataset"] = dataset_name
    return cat_counts

cat_cysec = categories_table(df_cysec, "CySec")
cat_alpaca = categories_table(df_alpaca, "Alpaca")
cat_all = pd.concat([cat_cysec, cat_alpaca], ignore_index=True)

# ----------------------------- Výstupy ----------------------------- #
outdir = ensure_outputs_dir()
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# Ulož obohacené CSV
df_cysec.to_csv(os.path.join(outdir, f"cysec_enriched_{ts}.csv"), index=False)
df_alpaca.to_csv(os.path.join(outdir, f"alpaca_enriched_{ts}.csv"), index=False)
cat_all.to_csv(os.path.join(outdir, f"categories_all_{ts}.csv"), index=False)

# ----------------------------- Koláč CySec vs Alpaca (červená vs modrá) ----------------------------- #
fig = plt.figure(figsize=(5.5, 5.5))
sizes = [len(df_cysec), len(df_alpaca)]
labels = [f"CySec (first {len(df_cysec)})", f"Alpaca (Next 620)"]
colors = [base_red, base_blue]
plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors)
plt.title("Proportion of Cysec vs Alpaca datasets")
save_fig(fig, os.path.join(outdir, f"pie_datasets_{ts}.png"))

# ----------------------------- Donut: názvy kategorií MIMO ----------------------------- #
# ----------------------------- Donut: inner labels (CySec/Alpaca), outer labels = category + global % ----------------------------- #
TOPN = 12
cysec_series = cat_cysec.set_index("category")["count"] if not cat_cysec.empty else pd.Series(dtype=int)
alpaca_series = cat_alpaca.set_index("category")["count"] if not cat_alpaca.empty else pd.Series(dtype=int)

def top_n_with_other(series_counts: pd.Series, n: int = 10) -> pd.Series:
    if series_counts.empty:
        return series_counts
    top = series_counts.nlargest(n)
    other = series_counts.drop(top.index).sum()
    if other > 0:
        top.loc["other"] = other
    return top

cysec_cats = top_n_with_other(cysec_series, TOPN)
alpaca_cats = top_n_with_other(alpaca_series, TOPN)

# outer ring values & labels
outer_sizes = list(cysec_cats.values) + list(alpaca_cats.values)
outer_names = list(cysec_cats.index) + list(alpaca_cats.index)

# global percentages across BOTH datasets (sum of all outer slices)
outer_total = sum(outer_sizes) if outer_sizes else 1
outer_labels = [f"{name}  {val/outer_total*100:.1f}%" for name, val in zip(outer_names, outer_sizes)]

# colors (lighter red tones for CySec categories, lighter blue for Alpaca)
outer_colors = [lighten_color(base_red, 0.50) for _ in cysec_cats] + \
               [lighten_color(base_blue, 0.50) for _ in alpaca_cats]

def plot_donut_categories_with_inner_dataset_labels():
    fig, ax = plt.subplots(figsize=(8.8, 7.4))

    # inner ring – datasets (no labels/autopct rendered; we'll place custom text)
    wedges1, _ = ax.pie(
        [len(df_cysec), len(df_alpaca)],
        radius=1.0,
        labels=None,
        colors=[base_red, base_blue],
        startangle=90,
        autopct=None,
        wedgeprops=dict(width=0.36, edgecolor="white")
    )

    # place dataset names + percentages INSIDE
    total_inner = len(df_cysec) + len(df_alpaca)
    share_cy = len(df_cysec) / total_inner * 100 if total_inner else 0
    share_al = len(df_alpaca) / total_inner * 100 if total_inner else 0

    ax.text(-0.35, 0.00, f"CySec\n{share_cy:.1f}%", ha="right", va="center",
            fontsize=12, fontweight="bold", color="#333")
    ax.text( 0.35, 0.00, f"Alpaca\n{share_al:.1f}%", ha="left", va="center",
            fontsize=12, fontweight="bold", color="#333")

    # outer ring – categories (labels outside)
    if outer_sizes:
        wedges2, _ = ax.pie(
            outer_sizes,
            radius=1.0,
            labels=None,
            colors=outer_colors,
            startangle=90,
            wedgeprops=dict(width=0.36, edgecolor="white")
        )

        # annotations outside with clearer leader lines + white background
        for i, (w, lab) in enumerate(zip(wedges2, outer_labels)):
            ang = (w.theta2 + w.theta1) / 2
            x, y = math.cos(math.radians(ang)), math.sin(math.radians(ang))

            r_start = 0.9                              
            r_text  = 1 if i % 2 == 0 else 1.30    

            xy = (x * r_start, y * r_start)
            jitter = 0.00 if y == 0 else (0.02 if y > 0 else -0.02)
            xytext = (x * r_text, y * r_text + jitter)

            ax.annotate(
                lab,
                xy=xy, xytext=xytext,
                ha=("left" if x >= 0 else "right"),
                va="center",
                fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
                arrowprops=dict(
                    arrowstyle="-",
                    color="#555",
                    lw=1.4,
                    shrinkA=0, shrinkB=5,
                    capstyle="round",
                    connectionstyle="arc3,rad=0.20"
                ),
            )

    ax.set(aspect="equal", title="Donut chart: datasets and their categories")
    return fig

fig = plot_donut_categories_with_inner_dataset_labels()
save_fig(fig, os.path.join(outdir, f"donut_datasets_categories_{ts}.png"))


# ----------------------------- Koláče kategorií (anti-overlap) ----------------------------- #
def plot_clean_pie(counts: pd.Series, title: str, path: str,
                   cmap_name: str = "tab20", min_pct_inside: float = 4.0,
                   angle_threshold_deg: float = 10.0):
    """
    Clean pie chart:
    - each category has a different color
    - small slices have alternating % labels (odd/even) placed at different distances
    - legend text fully visible (no cut-off brackets)
    """
    if counts.empty:
        return

    counts2 = counts.sort_values(ascending=False)
    values = counts2.values
    labels = counts2.index
    total = values.sum()

    cmap = colormaps.get_cmap(cmap_name)
    colors = [cmap(i / max(len(values)-1, 1)) for i in range(len(values))]

    fig, ax = plt.subplots(figsize=(10, 6))

    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        colors=colors,
        startangle=90,
        autopct=lambda p: f"{p:.0f}%" if p >= min_pct_inside else "",
        pctdistance=0.68,
        textprops={"color": "white", "weight": "bold", "fontsize": 11},
        wedgeprops=dict(edgecolor="white")
    )

    # For small slices: alternate label positions (odd/even)
    for i, (w, v) in enumerate(zip(wedges, values)):
        frac = v / total * 100.0
        theta = (w.theta2 - w.theta1)
        if frac < min_pct_inside or theta < angle_threshold_deg:
            ang = (w.theta2 + w.theta1) / 2
            x, y = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            # alternate distance for odd/even small labels
            offset = 1.10 if i % 2 == 0 else 1.20
            ax.annotate(
                f"{frac:.0f}%",
                xy=(x * 0.92, y * 0.92),
                xytext=(x * offset, y * offset),
                ha=("left" if x >= 0 else "right"),
                va="center",
                arrowprops=dict(arrowstyle="-", color="#999", lw=0.8, connectionstyle="arc3,rad=0.15"),
                fontsize=10, color="#444", weight="bold"
            )

    ax.set(aspect="equal")
    plt.title(title, fontsize=20, fontweight="bold", pad=12)

    # --- improved legend ---
    legend_labels = [f"{lab} ({int(val)})" for lab, val in zip(labels, values)]
    ax.legend(
        wedges, legend_labels,
        title="Categories",
        loc="center left",
        bbox_to_anchor=(1, 0.5),  # moved slightly right
        fontsize=11,
        frameon=False,
        handlelength=1.4
    )

    save_fig(fig, path)



# vykreslení „clean“ koláčů
plot_clean_pie(
    cat_cysec.set_index("category")["count"] if not cat_cysec.empty else pd.Series(dtype=int),
    "CySec – Category Distribution",
    os.path.join(outdir, f"pie_categories_cysec_{ts}.png"),
    cmap_name="tab20",
    min_pct_inside=4.0,     # >=4 % dovnitř, jinak ven s vodičem
    angle_threshold_deg=10.0
)
plot_clean_pie(
    cat_alpaca.set_index("category")["count"] if not cat_alpaca.empty else pd.Series(dtype=int),
    "Alpaca – Category Distribution",
    os.path.join(outdir, f"pie_categories_alpaca_{ts}.png"),
    cmap_name="tab20",
    min_pct_inside=4.0,
    angle_threshold_deg=10.0
)

# ----------------------------- Kybercrimes vs obecne_crimes ----------------------------- #
bucket_summary = (
    cat_all.groupby(["dataset", "bucket"], as_index=False)["count"]
    .sum()
    .rename(columns={"count":"total_count"})
)
bucket_summary["share"] = bucket_summary.groupby("dataset")["total_count"].transform(lambda x: x/x.sum())
bucket_summary.to_csv(os.path.join(outdir, f"bucket_summary_{ts}.csv"), index=False)

# bar porovnání
fig = plt.figure(figsize=(6,5))
for i, (name, group) in enumerate(bucket_summary.groupby("dataset")):
    xs = np.arange(len(group["bucket"])) + (i-0.5)*0.2
    color = base_red if name=="CySec" else base_blue
    plt.bar(xs, group["share"].values, width=0.4, label=name, color=color)
plt.xticks(np.arange(len(group["bucket"])), group["bucket"])
plt.ylabel("Podíl (share)")
plt.title("Podíl kybercrimes vs obecne_crimes")
plt.legend()
save_fig(fig, os.path.join(outdir, f"bar_buckets_share_{ts}.png"))

# ----------------------------- Top rozdíly kategorií ----------------------------- #
def cat_share(df_cat: pd.DataFrame, name: str) -> pd.Series:
    if df_cat.empty:
        return pd.Series(dtype=float)
    s = df_cat.set_index("category")["count"].astype(float)
    return s / s.sum()

share_cysec = cat_share(cat_cysec, "CySec")
share_alpaca = cat_share(cat_alpaca, "Alpaca")
diff = (share_cysec - share_alpaca).fillna(0).abs().sort_values(ascending=False)
top_diff = diff.head(15)
top_df = pd.DataFrame({
    "category": top_diff.index,
    "abs_diff_share": top_diff.values,
    "share_cysec": share_cysec.reindex(top_diff.index).fillna(0).values,
    "share_alpaca": share_alpaca.reindex(top_diff.index).fillna(0).values,
})
top_df.to_csv(os.path.join(outdir, f"top_category_diffs_{ts}.csv"), index=False)

# bar top rozdílů
fig = plt.figure(figsize=(10,6))
x = np.arange(len(top_df))
plt.bar(x-0.2, top_df["share_cysec"], width=0.4, label="CySec", color=base_red)
plt.bar(x+0.2, top_df["share_alpaca"], width=0.4, label="Alpaca", color=base_blue)
plt.xticks(x, top_df["category"], rotation=45, ha="right")
plt.ylabel("Podíl v datasetu")
plt.title("Top rozdíly kategorií mezi CySec a Alpaca")
plt.legend()
save_fig(fig, os.path.join(outdir, f"bar_top_category_diffs_{ts}.png"))

# ----------------------------- Statistika délek: describe + porovnání ----------------------------- #
def describe_lengths(df_part: pd.DataFrame, name: str) -> pd.DataFrame:
    cols = ["goal_len_chars","goal_len_words","trans_len_chars","trans_len_words",
            "ratio_trans_to_goal_chars","ratio_trans_to_goal_words"]
    return df_part[cols].describe(percentiles=[0.5,0.9,0.95]).T.assign(dataset=name)

desc_cysec = describe_lengths(df_cysec, "CySec")
desc_alpaca = describe_lengths(df_alpaca, "Alpaca")
desc_both = pd.concat([desc_cysec, desc_alpaca], axis=0)
desc_both.to_csv(os.path.join(outdir, f"lengths_describe_{ts}.csv"))

# histogram overlay (znaky/slova) originál
def overlay_hist(a, b, title, xlabel, path, bins=40):
    fig = plt.figure(figsize=(8,5))
    plt.hist(a, bins=bins, alpha=0.6, label=f"CySec ({len(df_cysec)})", color=base_red)
    plt.hist(b, bins=bins, alpha=0.6, label=f"Alpaca ({len(df_alpaca)})", color=base_blue)
    plt.xlabel(xlabel); plt.ylabel("Count"); plt.title(title); plt.legend()
    save_fig(fig, path)

overlay_hist(df_cysec["goal_len_chars"], df_alpaca["goal_len_chars"],
             "Distribuce délek – originál (znaky)",
             "Délka (znaky)", os.path.join(outdir, f"hist_goal_chars_overlay_{ts}.png"))
overlay_hist(df_cysec["goal_len_words"], df_alpaca["goal_len_words"],
             "Distribution of word count in original prompts",
             "Number of Words in Prompt", os.path.join(outdir, f"hist_goal_words_overlay_{ts}.png"), bins=20)

# histogram overlay pro překlady
overlay_hist(df_cysec["trans_len_chars"], df_alpaca["trans_len_chars"],
             "Distribuce délek – překlad (znaky)",
             "Délka (znaky)", os.path.join(outdir, f"hist_trans_chars_overlay_{ts}.png"))
overlay_hist(df_cysec["trans_len_words"], df_alpaca["trans_len_words"],
             "Distribution of word count in Chinese-translated prompts",
             "Number of Words in Prompt", os.path.join(outdir, f"hist_trans_words_overlay_{ts}.png"), bins=5)

# boxplot porovnání (originál i překlad)
def boxplot_compare(series_a, series_b, title, ylabel, path):
    fig = plt.figure(figsize=(6,5))
    plt.boxplot([series_a.dropna(), series_b.dropna()], labels=["CySec","Alpaca"])
    plt.ylabel(ylabel); plt.title(title)
    save_fig(fig, path)

boxplot_compare(df_cysec["goal_len_chars"], df_alpaca["goal_len_chars"],
                "Boxplot of lengths – original (characters)", "Length (characters)",
                os.path.join(outdir, f"box_goal_chars_{ts}.png"))

boxplot_compare(df_cysec["goal_len_words"], df_alpaca["goal_len_words"],
                "Boxplot of the number of words in original prompts", "Number of Words in Prompt",
                os.path.join(outdir, f"box_goal_words_{ts}.png"))

boxplot_compare(df_cysec["trans_len_chars"], df_alpaca["trans_len_chars"],
                "Boxplot of lengths – translation (characters)", "Length (characters)",
                os.path.join(outdir, f"box_trans_chars_{ts}.png"))

boxplot_compare(df_cysec["trans_len_words"], df_alpaca["trans_len_words"],
                "Boxplot of the number of words in Chinese-translated prompts", "Number of Words in Prompt",
                os.path.join(outdir, f"box_trans_words_{ts}.png"))


# scatter překlad vs originál (každý dataset zvlášť)
def scatter_trans_vs_goal(df_part, name, color, path):
    if df_part["goal_len_chars"].max() == 0 or df_part["trans_len_chars"].max() == 0:
        return
    fig = plt.figure(figsize=(6,6))
    plt.scatter(df_part["goal_len_chars"], df_part["trans_len_chars"], alpha=0.6, color=color)
    lim = max(df_part["goal_len_chars"].max(), df_part["trans_len_chars"].max())
    plt.plot([0, lim], [0, lim], linestyle="--")
    plt.xlabel("Originál – délka (znaky)")
    plt.ylabel("Překlad – délka (znaky)")
    plt.title(f"{name}: Překlad vs. originál (znaky)")
    save_fig(fig, path)

scatter_trans_vs_goal(df_cysec, "CySec", base_red, os.path.join(outdir, f"scatter_cysec_{ts}.png"))
scatter_trans_vs_goal(df_alpaca, "Alpaca", base_blue, os.path.join(outdir, f"scatter_alpaca_{ts}.png"))

# ----------------------------- Report ----------------------------- #
lines = []
lines.append(f"Vstupní CSV: {INPUT_CSV}")
lines.append(f"CySec (prvních {len(df_cysec)}): {len(df_cysec)} řádků")
lines.append(f"Alpaca (dalších {len(df_alpaca)}): {len(df_alpaca)} řádků")
lines.append("")
lines.append("Souhrn kybercrimes vs obecne_crimes (viz bucket_summary_*.csv a bar_buckets_share_*.png)")
if not cat_cysec.empty:
    top3_cysec = ", ".join((cat_cysec.sort_values('count', ascending=False)
                            .head(3)['category'] + " (" + cat_cysec.sort_values('count', ascending=False)
                            .head(3)['count'].astype(str) + ")").tolist())
    lines.append(f"Top kategorie CySec: {top3_cysec}")
if not cat_alpaca.empty:
    top3_alpaca = ", ".join((cat_alpaca.sort_values('count', ascending=False)
                             .head(3)['category'] + " (" + cat_alpaca.sort_values('count', ascending=False)
                             .head(3)['count'].astype(str) + ")").tolist())
    lines.append(f"Top kategorie Alpaca: {top3_alpaca}")
lines.append("")
lines.append("Popisné statistiky délek: lengths_describe_*.csv")
lines.append("Koláče:")
lines.append(" - pie_datasets_*.png (CySec vs Alpaca)")
lines.append(" - donut_datasets_categories_*.png (vnitřní dataset, vnější kategorie — názvy mimo)")
lines.append(" - pie_categories_cysec_*.png, pie_categories_alpaca_*.png (anti-overlap: malé % ven)")
lines.append("Porovnání kategorií: bar_top_category_diffs_*.png + top_category_diffs_*.csv")
lines.append("Histogramy overlay + boxploty: hist_*_overlay_*.png, box_*_*.png")
report = "\n".join(lines)
with open(os.path.join(outdir, f"report_{ts}.txt"), "w", encoding="utf-8") as f:
    f.write(report)

print("\n✅ Hotovo! Výstupy najdeš ve složce 'outputs/'.\n")
print(report)



# ----------------------------- Config ----------------------------- #
INPUT_CSV = "/storage/brno2/home/xkaska01/master/my_implementation/dataset/base_dataset.csv"
CYSEC_N = 500
ALPACA_N = 619

BASE_RED = "#e41a1c"
BASE_BLUE = "#377eb8"

# ----------------------------- Helpers ----------------------------- #
def ensure_outputs_dir(path="outputs"):
    os.makedirs(path, exist_ok=True)
    return path

def save_fig(fig, path):
    try:
        fig.tight_layout()
    except Exception:
        plt.subplots_adjust(left=0.06, right=0.88, top=0.92, bottom=0.06)
    fig.savefig(path, dpi=150)
    plt.close(fig)

def normalize_label(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())

def explode_multilabels(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=str)
    tmp = series.fillna("").astype(str)
    tmp = tmp.str.replace(r"[;|/]", ",", regex=True)
    out = tmp.str.split(",").explode().map(normalize_label)
    return out[out != ""]

def lighten_color(hex_color, factor=0.5):
    """Lighten HEX color toward white by `factor`."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"

# ----------------------------- Load ----------------------------- #
if not os.path.isfile(INPUT_CSV):
    print(f"❌ CSV not found: {INPUT_CSV}")
    sys.exit(1)

df = pd.read_csv(INPUT_CSV)
if len(df) < CYSEC_N + ALPACA_N:
    print(f"⚠️ CSV has {len(df)} rows; splitting as first {CYSEC_N} = CySec, rest = Alpaca.")

df_cysec = df.iloc[:CYSEC_N].copy()
df_alpaca = df.iloc[CYSEC_N:CYSEC_N+ALPACA_N].copy()

# ----------------------------- Categories ----------------------------- #
def categories_counts(df_part: pd.DataFrame) -> pd.Series:
    if "target" not in df_part.columns:
        return pd.Series(dtype=int)
    labels = explode_multilabels(df_part["target"])
    return labels.value_counts()

cats_cysec = categories_counts(df_cysec)
cats_alpaca = categories_counts(df_alpaca)

# ----------------------------- Clean pie (anti-overlap for tiny slices) ----------------------------- #
def plot_clean_pie(ax, counts: pd.Series, title: str, cmap_name="tab20",
                   min_pct_inside: float = 4.0, angle_threshold_deg: float = 10.0,
                   legend: bool = True):
    """
    - colors from cmap
    - big slices: % inside
    - small slices: % outside with leader line (alternating distance to reduce overlap)
    - category names in legend (right of the axis) if legend=True
    """
    if counts.empty:
        ax.set_axis_off()
        ax.set_title(title, fontsize=16, fontweight="bold", pad=8)
        return

    counts2 = counts.sort_values(ascending=False)
    values = counts2.values
    labels = counts2.index
    total = values.sum()

    cmap = colormaps.get_cmap(cmap_name)
    colors = [cmap(i / max(len(values)-1, 1)) for i in range(len(values))]

    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        colors=colors,
        startangle=90,
        autopct=lambda p: f"{p:.0f}%" if p >= min_pct_inside else "",
        pctdistance=0.68,
        textprops={"color": "white", "weight": "bold", "fontsize": 11},
        wedgeprops=dict(edgecolor="white")
    )

    # small slices: put % outside, alternating distances (even=closer, odd=further)
    for i, (w, v) in enumerate(zip(wedges, values)):
        frac = v / total * 100.0
        theta = (w.theta2 - w.theta1)
        if frac < min_pct_inside or theta < angle_threshold_deg:
            ang = (w.theta2 + w.theta1) / 2
            x, y = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            r1 = 1.15 if (i % 2 == 0) else 1.30
            ax.annotate(
                f"{frac:.0f}%",
                xy=(x * 0.92, y * 0.92),
                xytext=(x * r1, y * r1),
                ha=("left" if x >= 0 else "right"),
                va="center",
                arrowprops=dict(arrowstyle="-", color="#777", lw=1.2,
                                connectionstyle="arc3,rad=0.18", capstyle="round"),
                fontsize=10, color="#333", weight="bold"
            )

    ax.set(aspect="equal")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=8)

    if legend:
        legend_labels = [f"{lab}  ({int(val)})" for lab, val in zip(labels, values)]
        ax.legend(
            wedges, legend_labels, title="Categories",
            loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=False, handlelength=1.4
        )

# ----------------------------- Panel (3 pies in one) ----------------------------- #
def plot_panel_pies(df_cysec, df_alpaca, cats_cysec, cats_alpaca, out_path: str):
    # figure layout: 1 row x 2 columns grid, right column split into 2 rows
    fig = plt.figure(figsize=(14, 7.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.6], height_ratios=[1, 1], wspace=0.20, hspace=0.25)

    # LEFT: dataset share
    ax_left = fig.add_subplot(gs[:, 0])
    sizes = [len(df_cysec), len(df_alpaca)]
    labels = [f"CySec (first {len(df_cysec)})", f"Alpaca (next {len(df_alpaca)})"]
    colors = [BASE_RED, BASE_BLUE]
    ax_left.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors,
                textprops={"fontsize": 12})
    ax_left.set_title("Dataset share (CySec vs Alpaca)", fontsize=18, fontweight="bold", pad=10)
    ax_left.set_aspect("equal")

    # TOP-RIGHT: CySec categories
    ax_tr = fig.add_subplot(gs[0, 1])
    plot_clean_pie(ax_tr, cats_cysec, "CySec – category distribution", cmap_name="tab20",
                   min_pct_inside=4.0, angle_threshold_deg=10.0, legend=False)

    # BOTTOM-RIGHT: Alpaca categories
    ax_br = fig.add_subplot(gs[1, 1])
    plot_clean_pie(ax_br, cats_alpaca, "Alpaca – category distribution", cmap_name="tab20",
                   min_pct_inside=4.0, angle_threshold_deg=10.0, legend=False)

    # make one combined legend for both category pies (unique labels from Alpaca -> same palette mapping is okay visually)
    if not cats_alpaca.empty:
        counts2 = cats_alpaca.sort_values(ascending=False)
        values = counts2.values
        labels = counts2.index
        cmap = colormaps.get_cmap("tab20")
        colors = [cmap(i / max(len(values)-1, 1)) for i in range(len(values))]
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10)
                   for c in colors]
        fig.legend(handles, [f"{lab} ({int(val)})" for lab, val in zip(labels, values)],
                   title="Categories (counts)", loc="center right", bbox_to_anchor=(0.98, 0.5),
                   fontsize=10, frameon=False)

    save_fig(fig, out_path)

# ----------------------------- Run ----------------------------- #
outdir = ensure_outputs_dir()
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save enriched counts for reference
cats_cysec.reset_index().rename(columns={"index": "category", 0: "count"}).to_csv(
    os.path.join(outdir, f"cysec_categories_{ts}.csv"), index=False
)
cats_alpaca.reset_index().rename(columns={"index": "category", 0: "count"}).to_csv(
    os.path.join(outdir, f"alpaca_categories_{ts}.csv"), index=False
)

# Render the triple-pie panel
panel_path = os.path.join(outdir, f"panel_pies_{ts}.png")
plot_panel_pies(df_cysec, df_alpaca, cats_cysec, cats_alpaca, panel_path)

print("✅ Done. Main figure:", panel_path)
print("   Extra CSVs saved in:", outdir)