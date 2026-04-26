import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# =========================
# NASTAVENÍ
# =========================
DATA_FOLDER = "."
OUTPUT_DIR = "./plots_agreement"
FILES_BENIGN = sorted(glob.glob(os.path.join(DATA_FOLDER, "out_*_benign.csv")))
FILES_HARMFUL = sorted(glob.glob(os.path.join(DATA_FOLDER, "out_*_harmful.csv")))

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def find_score_column(df, candidates, file_name, required=True):
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

def score_to_band(score):
    if pd.isna(score):
        return np.nan

    score = float(score)

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

def band_label(code):
    mapping = {
        0: "0",
        1: "1-2",
        2: "3-5",
        3: "6-7",
        4: "8-9",
        5: "10"
    }
    return mapping.get(code, "NaN")

def round_mean_band(series):
    rounded = np.rint(series).astype(float)
    rounded = np.clip(rounded, 0, 5)
    return pd.Series(rounded, index=series.index).astype("Int64")

def load_group_data(file_list, group_name="GROUP"):
    all_scores_raw = []
    judge_scores_raw_list = []

    for f_path in file_list:
        file_name = os.path.basename(f_path)

        try:
            df = pd.read_csv(f_path)
            df = normalize_columns(df)

            human_col = find_score_column(
                df,
                candidates=["human_score", "human score", "score", "human"],
                file_name=file_name
            )

            judge_col = find_score_column(
                df,
                candidates=["judge_model_score", "judge score", "judge_model", "judge"],
                file_name=file_name
            )

            s_human_raw = pd.to_numeric(df[human_col], errors="coerce").astype(float)
            s_judge_raw = pd.to_numeric(df[judge_col], errors="coerce").astype(float)

            all_scores_raw.append(s_human_raw)
            judge_scores_raw_list.append(s_judge_raw)

        except Exception as e:
            print(f"❌ Chyba v souboru {file_name}: {e}")

    if not all_scores_raw:
        print(f"❌ Pro skupinu {group_name} nebyl načten žádný validní soubor.")
        return None

    human_matrix_raw = pd.DataFrame(all_scores_raw).T.astype(float)
    human_matrix_raw.columns = [f"Annotator_{i+1}" for i in range(len(all_scores_raw))]

    judge_scores_raw = pd.DataFrame(judge_scores_raw_list).T.mean(axis=1)

    clean_mask = human_matrix_raw.notna().all(axis=1) & judge_scores_raw.notna()

    human_matrix_raw = human_matrix_raw[clean_mask].reset_index(drop=True)
    judge_scores_raw = judge_scores_raw[clean_mask].reset_index(drop=True)

    human_consensus_raw = human_matrix_raw.median(axis=1)

    human_matrix_band = human_matrix_raw.map(score_to_band)
    human_consensus_band = round_mean_band(human_matrix_band.median(axis=1))
    judge_scores_band = judge_scores_raw.apply(score_to_band)
    judge_scores_band = round_mean_band(judge_scores_band)

    abs_diff_raw = (judge_scores_raw - human_consensus_raw).abs()

    return {
        "group_name": group_name,
        "human_matrix_raw": human_matrix_raw,
        "judge_scores_raw": judge_scores_raw,
        "human_consensus_raw": human_consensus_raw,
        "human_matrix_band": human_matrix_band,
        "human_consensus_band": pd.Series(human_consensus_band),
        "judge_scores_band": pd.Series(judge_scores_band),
        "abs_diff_raw": abs_diff_raw
    }

# =========================
# GRAFY
# =========================
def plot_confusion_heatmap(human_consensus_band, judge_scores_band, group_name, output_dir):
    labels = [0, 1, 2, 3, 4, 5]

    df_cm = pd.DataFrame({
        "human": pd.to_numeric(human_consensus_band, errors="coerce"),
        "judge": pd.to_numeric(judge_scores_band, errors="coerce")
    }).dropna()

    if df_cm.empty:
        print(f"❌ {group_name}: pro confusion matrix nejsou žádná validní data.")
        return

    human_clean = df_cm["human"].astype(int)
    judge_clean = df_cm["judge"].astype(int)

    cm = confusion_matrix(human_clean, judge_clean, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([band_label(x) for x in labels])
    ax.set_yticklabels([band_label(x) for x in labels])

    ax.set_xlabel("Judge Score (Bands)")
    ax.set_ylabel("Human Consensus (Bands)")
    ax.set_title(f"{group_name}: Confusion Matrix / Heatmap")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{group_name.lower()}_confusion_heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Uložen graf: {save_path}")
    plt.show()
    plt.close()

def plot_score_distribution(human_matrix_raw, human_consensus_raw, judge_scores_raw, group_name, output_dir):
    all_human_scores = human_matrix_raw.stack().dropna().values
    consensus_scores = human_consensus_raw.dropna().values
    judge_scores = judge_scores_raw.dropna().values

    bins = np.arange(-0.5, 11.5, 1)

    plt.figure(figsize=(12, 7))
    plt.hist(all_human_scores, bins=bins, alpha=0.40, label="Všechny lidské anotace", edgecolor="black", density=True)
    plt.hist(consensus_scores, bins=bins, alpha=0.50, label="Human consensus", edgecolor="black", density=True)
    plt.hist(judge_scores, bins=bins, alpha=0.50, label="Judge", edgecolor="black", density=True)

    plt.xticks(range(0, 11))
    plt.xlabel("Skóre 0-10")
    plt.ylabel("Relativní četnost")
    plt.title(f"{group_name}: Distribuce skóre")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{group_name.lower()}_score_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Uložen graf: {save_path}")
    plt.show()
    plt.close()

def plot_bland_altman(human_consensus_raw, judge_scores_raw, group_name, output_dir):
    mean_scores = (human_consensus_raw + judge_scores_raw) / 2
    diff_scores = judge_scores_raw - human_consensus_raw

    mean_diff = diff_scores.mean()
    std_diff = diff_scores.std(ddof=1)

    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    plt.figure(figsize=(10, 6))
    plt.scatter(mean_scores, diff_scores, alpha=0.6)
    plt.axhline(mean_diff, linestyle="--", label=f"Mean diff = {mean_diff:.2f}")
    plt.axhline(loa_upper, linestyle="--", label=f"+1.96 SD = {loa_upper:.2f}")
    plt.axhline(loa_lower, linestyle="--", label=f"-1.96 SD = {loa_lower:.2f}")

    plt.xlabel("Průměr (Human Consensus + Judge) / 2")
    plt.ylabel("Rozdíl (Judge - Human Consensus)")
    plt.title(f"{group_name}: Bland–Altman graf")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{group_name.lower()}_bland_altman.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Uložen graf: {save_path}")
    plt.show()
    plt.close()

def plot_abs_diff_boxplot(abs_diff_benign, abs_diff_harmful, output_dir):
    data = [abs_diff_benign.dropna().values, abs_diff_harmful.dropna().values]

    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=["BENIGN", "HARMFUL"])
    plt.ylabel("|Judge - Human Consensus|")
    plt.title("Boxplot absolutních odchylek")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "benign_vs_harmful_abs_diff_boxplot.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Uložen graf: {save_path}")
    plt.show()
    plt.close()

def plot_agreement_vs_tolerance(human_consensus_raw, judge_scores_raw, group_name, output_dir, max_tolerance=3):
    tolerances = list(range(0, max_tolerance + 1))
    abs_diff = (judge_scores_raw - human_consensus_raw).abs()

    agreement = [np.mean(abs_diff <= t) * 100 for t in tolerances]

    plt.figure(figsize=(8, 6))
    plt.plot(tolerances, agreement, marker="o")
    plt.xticks(tolerances)
    plt.ylim(0, 100)
    plt.xlabel("Tolerance (body)")
    plt.ylabel("Shoda (%)")
    plt.title(f"{group_name}: Agreement vs. Tolerance")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{group_name.lower()}_agreement_vs_tolerance.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Uložen graf: {save_path}")
    plt.show()
    plt.close()

# =========================
# HLAVNÍ SPUŠTĚNÍ
# =========================
benign_data = load_group_data(FILES_BENIGN, "BENIGN") if FILES_BENIGN else None
harmful_data = load_group_data(FILES_HARMFUL, "HARMFUL") if FILES_HARMFUL else None

if benign_data is not None:
    plot_confusion_heatmap(
        benign_data["human_consensus_band"],
        benign_data["judge_scores_band"],
        "BENIGN",
        OUTPUT_DIR
    )
    plot_score_distribution(
        benign_data["human_matrix_raw"],
        benign_data["human_consensus_raw"],
        benign_data["judge_scores_raw"],
        "BENIGN",
        OUTPUT_DIR
    )
    plot_bland_altman(
        benign_data["human_consensus_raw"],
        benign_data["judge_scores_raw"],
        "BENIGN",
        OUTPUT_DIR
    )
    plot_agreement_vs_tolerance(
        benign_data["human_consensus_raw"],
        benign_data["judge_scores_raw"],
        "BENIGN",
        OUTPUT_DIR
    )

if harmful_data is not None:
    plot_confusion_heatmap(
        harmful_data["human_consensus_band"],
        harmful_data["judge_scores_band"],
        "HARMFUL",
        OUTPUT_DIR
    )
    plot_score_distribution(
        harmful_data["human_matrix_raw"],
        harmful_data["human_consensus_raw"],
        harmful_data["judge_scores_raw"],
        "HARMFUL",
        OUTPUT_DIR
    )
    plot_bland_altman(
        harmful_data["human_consensus_raw"],
        harmful_data["judge_scores_raw"],
        "HARMFUL",
        OUTPUT_DIR
    )
    plot_agreement_vs_tolerance(
        harmful_data["human_consensus_raw"],
        harmful_data["judge_scores_raw"],
        "HARMFUL",
        OUTPUT_DIR
    )

if benign_data is not None and harmful_data is not None:
    plot_abs_diff_boxplot(
        benign_data["abs_diff_raw"],
        harmful_data["abs_diff_raw"],
        OUTPUT_DIR
    )