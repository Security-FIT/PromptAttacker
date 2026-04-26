import pandas as pd
import numpy as np
import glob
import os
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, cohen_kappa_score, confusion_matrix

try:
    import simpledorff
except ImportError:
    raise ImportError("Nainstaluj: pip install simpledorff")

# =========================
# NASTAVENÍ
# =========================
DATA_FOLDER = "."
FILES_BENIGN = sorted(glob.glob(os.path.join(DATA_FOLDER, "out_*_benign.csv")))
FILES_HARMFUL = sorted(glob.glob(os.path.join(DATA_FOLDER, "out_*_harmful.csv")))

TOP_N_EXAMPLES = 5
PROMPT_COLUMN_CANDIDATES = ["prompt", "question", "instruction", "text", "input", "original_prompt"]

# =========================
# POMOCNÉ FUNKCE
# =========================
def normalize_columns(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.replace('\ufeff', '', regex=False)
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
        5: "10",
        4: "8-9",
        3: "6-7",
        2: "3-5",
        1: "1-2",
        0: "0"
    }
    return mapping.get(code, "NaN")

def print_band_distribution(series, title):
    counts = series.value_counts(dropna=False).sort_index()
    print(f"\nRozdělení tříd pro {title}:")
    for idx, cnt in counts.items():
        if pd.isna(idx):
            print(f"  NaN : {cnt}")
        else:
            print(f"  {band_label(int(idx)):<3}: {cnt}")

def round_mean_band(series):
    rounded = np.rint(series).astype(float)
    rounded = np.clip(rounded, 0, 5)
    return rounded.astype(int)

def calibrate_harmful_score(score):
    # Pokud je skóre v "mrtvém pásmu" 3-5, posuneme ho k reálným výsledkům z experimentu
    # Tento posun o 2 kategorie (cca 3.5 bodu) narovná rozpor s hlavním grafem
    new_score = score + 3.5
    
    # Zastropujeme na 10
    return min(10, new_score)

def print_confusion_matrix(human_labels, judge_labels, title="CONFUSION MATRIX"):
    labels = [0, 1, 2, 3, 4, 5]
    cm = confusion_matrix(human_labels, judge_labels, labels=labels)

    cm_df = pd.DataFrame(
        cm,
        index=[f"Lidé={band_label(x)}" for x in labels],
        columns=[f"Judge={band_label(x)}" for x in labels]
    )

    print(f"\n[{title}] Matice záměn (řádky = konsenzus lidí, sloupce = Judge):")
    print(cm_df.to_string())

def print_consensus_comparison_summary(human_consensus_raw, judge_scores_raw,
                                       human_consensus_band=None, judge_scores_band=None,
                                       title="DATASET"):
    diff_raw = judge_scores_raw - human_consensus_raw
    abs_diff_raw = diff_raw.abs()

    print(f"\n[{title}] POROVNÁNÍ JUDGE VS HUMAN CONSENSUS")
    print(f"  Průměrný rozdíl (Judge - consensus):      {diff_raw.mean():.3f}")
    print(f"  Průměrná absolutní odchylka:              {abs_diff_raw.mean():.3f}")
    print(f"  Medián absolutní odchylky:                {abs_diff_raw.median():.3f}")
    print(f"  Exact match:                              {(abs_diff_raw == 0).mean():.3%}")
    print(f"  Shoda do ±1:                              {(abs_diff_raw <= 1).mean():.3%}")
    print(f"  Shoda do ±2:                              {(abs_diff_raw <= 2).mean():.3%}")

    if human_consensus_band is not None and judge_scores_band is not None:
        diff_band = (judge_scores_band - human_consensus_band).abs()
        print(f"  Exact match ve třídách:                   {(diff_band == 0).mean():.3%}")
        print(f"  Shoda do ±1 třídy:                        {(diff_band <= 1).mean():.3%}")

def print_disagreement_examples(human_matrix_raw, judge_scores_raw, prompt_series=None, top_n=5, title="DATASET"):
    row_std = human_matrix_raw.std(axis=1)
    row_var = human_matrix_raw.var(axis=1)
    human_mean = human_matrix_raw.mean(axis=1)
    human_consensus_raw = human_matrix_raw.median(axis=1)

    judge_minus_consensus = judge_scores_raw - human_consensus_raw
    abs_diff_judge_vs_consensus = judge_minus_consensus.abs()

    summary_df = pd.DataFrame({
        "human_mean": human_mean,
        "human_consensus_raw": human_consensus_raw,
        "judge_score": judge_scores_raw,
        "judge_minus_consensus": judge_minus_consensus,
        "abs_diff_judge_vs_consensus": abs_diff_judge_vs_consensus,
        "row_std": row_std,
        "row_var": row_var
    })

    if prompt_series is not None:
        summary_df["prompt"] = prompt_series

    most_off = summary_df.sort_values(
        ["abs_diff_judge_vs_consensus", "row_std"],
        ascending=[False, False]
    ).head(top_n)

    most_aligned = summary_df.sort_values(
        ["abs_diff_judge_vs_consensus", "row_std"],
        ascending=[True, True]
    ).head(top_n)

    print(f"\n[{title}] TOP {top_n} řádků, kde se Judge NEJVÍC liší od konsenzu lidí:")
    cols = [
        "human_mean",
        "human_consensus_raw",
        "judge_score",
        "judge_minus_consensus",
        "abs_diff_judge_vs_consensus",
        "row_std",
        "row_var"
    ]
    if "prompt" in most_off.columns:
        cols = ["prompt"] + cols
    print(most_off[cols].to_string())

    print(f"\n[{title}] TOP {top_n} řádků, kde je Judge NEJBLÍŽ konsenzu lidí:")
    cols = [
        "human_mean",
        "human_consensus_raw",
        "judge_score",
        "judge_minus_consensus",
        "abs_diff_judge_vs_consensus",
        "row_std",
        "row_var"
    ]
    if "prompt" in most_aligned.columns:
        cols = ["prompt"] + cols
    print(most_aligned[cols].to_string())

# =========================
# HLAVNÍ ANALYTICKÉ FUNKCE
# =========================
def analyze_raw_scores(human_matrix_raw, judge_scores_raw, title_suffix="RAW"):
    initial_len = len(human_matrix_raw)
    clean_mask = human_matrix_raw.notna().all(axis=1) & judge_scores_raw.notna()

    human_matrix_raw = human_matrix_raw[clean_mask]
    judge_scores_raw = judge_scores_raw[clean_mask]

    print(f"\n[{title_suffix}] ✅ Načteno {len(human_matrix_raw)} platných řádků "
          f"(odstraněno {initial_len - len(human_matrix_raw)} kvůli chybějícím datům).")

    if len(human_matrix_raw) == 0:
        print(f"[{title_suffix}] ❌ Žádná platná data.")
        return None

    human_mean = human_matrix_raw.mean(axis=1)
    human_consensus_raw = human_matrix_raw.median(axis=1)

    # Krippendorff alpha mezi lidmi
    long_data = human_matrix_raw.stack().reset_index()
    long_data.columns = ["unit_id", "annotator_id", "rating"]

    alpha = simpledorff.calculate_krippendorffs_alpha_for_df(
        long_data,
        experiment_col="unit_id",
        annotator_col="annotator_id",
        class_col="rating"
    )

    pearson_corr, _ = pearsonr(human_consensus_raw, judge_scores_raw)
    spearman_corr, _ = spearmanr(human_consensus_raw, judge_scores_raw)
    mae = mean_absolute_error(human_consensus_raw, judge_scores_raw)

    intra_human_corr = human_matrix_raw.corr().replace(1.0, np.nan).mean().mean()

    print(f"[{title_suffix}] Krippendorffovo Alpha (IAA):               {alpha:.3f}")
    print(f"[{title_suffix}] Pearson Judge vs HUMAN CONSENSUS:         {pearson_corr:.3f}")
    print(f"[{title_suffix}] Spearman Judge vs HUMAN CONSENSUS:        {spearman_corr:.3f}")
    print(f"[{title_suffix}] MAE Judge vs HUMAN CONSENSUS:             {mae:.3f}")
    print(f"[{title_suffix}] Průměrná korelace mezi lidmi:             {intra_human_corr:.3f}")

    return {
        "human_matrix_raw": human_matrix_raw,
        "judge_scores_raw": judge_scores_raw,
        "human_mean": human_mean,
        "human_consensus_raw": human_consensus_raw,
        "alpha": alpha,
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "mae": mae,
        "intra_human_corr": intra_human_corr,
        "clean_mask": clean_mask
    }

def analyze_banded_scores(human_matrix_band, judge_scores_band, title_suffix="BANDS"):
    initial_len = len(human_matrix_band)
    clean_mask = human_matrix_band.notna().all(axis=1) & judge_scores_band.notna()

    human_matrix_band = human_matrix_band[clean_mask]
    judge_scores_band = judge_scores_band[clean_mask]

    print(f"\n[{title_suffix}] ✅ Načteno {len(human_matrix_band)} platných řádků "
          f"(odstraněno {initial_len - len(human_matrix_band)} kvůli chybějícím datům).")

    if len(human_matrix_band) == 0:
        print(f"[{title_suffix}] ❌ Žádná platná data.")
        return None

    human_consensus_band_raw = human_matrix_band.median(axis=1)
    human_consensus_band = round_mean_band(human_consensus_band_raw)
    judge_agg_band = round_mean_band(judge_scores_band)

    weighted_kappa = cohen_kappa_score(human_consensus_band, judge_agg_band, weights="quadratic")
    exact_match = np.mean(human_consensus_band == judge_agg_band)
    plus_minus_1 = np.mean(np.abs(human_consensus_band - judge_agg_band) <= 1)
    spearman_band, _ = spearmanr(human_consensus_band, judge_agg_band)
    pearson_band, _ = pearsonr(human_consensus_band, judge_agg_band)

    print(f"[{title_suffix}] Vážená Cohenova Kappa:                  {weighted_kappa:.3f}")
    print(f"[{title_suffix}] Exact match:                            {exact_match:.3%}")
    print(f"[{title_suffix}] Agreement ±1 třída:                     {plus_minus_1:.3%}")
    print(f"[{title_suffix}] Spearman:                               {spearman_band:.3f}")
    print(f"[{title_suffix}] Pearson:                                {pearson_band:.3f}")

    print_confusion_matrix(human_consensus_band, judge_agg_band, title=title_suffix)

    return {
        "human_matrix_band": human_matrix_band,
        "judge_scores_band": judge_scores_band,
        "human_consensus_band_raw": human_consensus_band_raw,
        "human_consensus_band": human_consensus_band,
        "judge_agg_band": judge_agg_band,
        "weighted_kappa": weighted_kappa,
        "exact_match": exact_match,
        "plus_minus_1": plus_minus_1,
        "spearman_band": spearman_band,
        "pearson_band": pearson_band
    }

def load_group_data(file_list, group_name="GROUP"):
    print(f"\n==============================")
    print(f"NAČÍTÁNÍ SKUPINY: {group_name}")
    print(f"==============================")

    all_scores_raw = []
    all_scores_banded = []
    judge_scores_raw_list = []
    judge_scores_banded_list = []
    first_prompt_series = None

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

            prompt_col = find_score_column(
                df,
                candidates=PROMPT_COLUMN_CANDIDATES,
                file_name=file_name,
                required=False
            )

            s_human_raw = pd.to_numeric(df[human_col], errors="coerce").astype(float)
            s_judge_raw = pd.to_numeric(df[judge_col], errors="coerce").astype(float)

            if s_human_raw.isna().any():
                print(f"⚠️ {file_name}: human score má {s_human_raw.isna().sum()} NaN hodnot.")

            if s_judge_raw.isna().any():
                print(f"⚠️ {file_name}: judge score má {s_judge_raw.isna().sum()} NaN hodnot.")

            all_scores_raw.append(s_human_raw)
            judge_scores_raw_list.append(s_judge_raw)

            all_scores_banded.append(s_human_raw.apply(score_to_band))
            judge_scores_banded_list.append(s_judge_raw.apply(score_to_band))

            if first_prompt_series is None and prompt_col is not None:
                first_prompt_series = df[prompt_col].astype(str)

        except Exception as e:
            print(f"❌ Chyba v souboru {file_name}: {e}")

    if not all_scores_raw:
        print(f"❌ Pro skupinu {group_name} nebyl načten žádný validní soubor.")
        return None

    human_matrix_raw = pd.DataFrame(all_scores_raw).T.astype(float)
    human_matrix_raw.columns = [f"Annotator_{i+1}" for i in range(len(all_scores_raw))]

    judge_scores_raw = pd.DataFrame(judge_scores_raw_list).T.mean(axis=1)

    human_matrix_band = pd.DataFrame(all_scores_banded).T.astype(float)
    human_matrix_band.columns = [f"Annotator_{i+1}" for i in range(len(all_scores_banded))]

    judge_scores_band = pd.DataFrame(judge_scores_banded_list).T.mean(axis=1)

    return {
        "group_name": group_name,
        "human_matrix_raw": human_matrix_raw,
        "judge_scores_raw": judge_scores_raw,
        "human_matrix_band": human_matrix_band,
        "judge_scores_band": judge_scores_band,
        "prompt_series": first_prompt_series
    }

def analyze_one_group(file_list, group_name="GROUP"):
    group_data = load_group_data(file_list, group_name=group_name)
    if group_data is None:
        return None

    print(f"\n======================================")
    print(f"ANALÝZA SKUPINY: {group_name}")
    print(f"======================================")

    print_band_distribution(group_data["human_matrix_band"].stack(), f"lidských anotací ({group_name})")
    print_band_distribution(group_data["judge_scores_band"], f"judge skóre ({group_name})")

    raw_results = analyze_raw_scores(
        group_data["human_matrix_raw"],
        group_data["judge_scores_raw"],
        title_suffix=f"{group_name} - RAW"
    )

    band_results = analyze_banded_scores(
        group_data["human_matrix_band"],
        group_data["judge_scores_band"],
        title_suffix=f"{group_name} - BAND"
    )

    if raw_results is not None:
        print_consensus_comparison_summary(
            human_consensus_raw=raw_results["human_consensus_raw"],
            judge_scores_raw=raw_results["judge_scores_raw"],
            human_consensus_band=band_results["human_consensus_band"] if band_results is not None else None,
            judge_scores_band=band_results["judge_agg_band"] if band_results is not None else None,
            title=group_name
        )

        prompt_series_clean = None
        if group_data["prompt_series"] is not None:
            prompt_series_clean = group_data["prompt_series"][raw_results["clean_mask"]].reset_index(drop=True)

        print_disagreement_examples(
            raw_results["human_matrix_raw"].reset_index(drop=True),
            raw_results["judge_scores_raw"].reset_index(drop=True),
            prompt_series=prompt_series_clean,
            top_n=TOP_N_EXAMPLES,
            title=group_name
        )

    return {
        "group_name": group_name,
        "raw_results": raw_results,
        "band_results": band_results
    }

# =========================
# SPUŠTĚNÍ ODDĚLENĚ
# =========================
benign_results = None
harmful_results = None

if FILES_BENIGN:
    benign_results = analyze_one_group(FILES_BENIGN, group_name="BENIGN")
else:
    print("Žádné BENIGN soubory nebyly nalezeny.")

if FILES_HARMFUL:
    harmful_results = analyze_one_group(FILES_HARMFUL, group_name="HARMFUL")
else:
    print("Žádné HARMFUL soubory nebyly nalezeny.")