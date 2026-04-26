import sys
import re
import math
import pandas as pd
import matplotlib.pyplot as plt


def find_judge_column(df: pd.DataFrame) -> str:
    candidates = [col for col in df.columns if "judge" in col.lower()]
    if not candidates:
        raise ValueError(
            f"Nenašel jsem žádný sloupec obsahující 'judge'. Dostupné sloupce: {list(df.columns)}"
        )
    if len(candidates) > 1:
        print(f"Nalezeno více judge sloupců, používám první: {candidates[0]}")
    return candidates[0]


def find_human_column(df: pd.DataFrame) -> str:
    candidates = [col for col in df.columns if col.lower() == "human_score"]
    if not candidates:
        candidates = [col for col in df.columns if "human_score" in col.lower()]

    if not candidates:
        raise ValueError(
            f"Nenašel jsem sloupec 'human_score'. Dostupné sloupce: {list(df.columns)}"
        )

    if len(candidates) > 1:
        print(f"Nalezeno více human_score sloupců, používám první: {candidates[0]}")

    return candidates[0]


def parse_score(value):
    if pd.isna(value):
        return None

    text = str(value).strip()

    if re.fullmatch(r"(10|[0-9])", text):
        return int(text)

    match = re.search(r"\b(10|[0-9])\b", text)
    if match:
        return int(match.group(1))

    return None


def prepare_counts(series: pd.Series) -> pd.Series:
    scores = series.apply(parse_score).dropna().astype(int)
    scores = scores[(scores >= 0) & (scores <= 10)]
    return scores.value_counts().reindex(range(11), fill_value=0)


def main():
    if len(sys.argv) < 6:
        print("Použití: python plot_scores_compare.py soubor1.csv soubor2.csv soubor3.csv soubor4.csv soubor5.csv")
        sys.exit(1)

    csv_files = sys.argv[1:6]
    num_files = len(csv_files)

    cols = 2
    rows = math.ceil(num_files / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = axes.flatten()

    x = list(range(11))
    width = 0.4
    y_max = 450

    for i, csv_path in enumerate(csv_files):
        df = pd.read_csv(csv_path)

        judge_col = find_judge_column(df)
        human_col = find_human_column(df)

        print(f"[{csv_path}] Používám judge sloupec: {judge_col}")
        print(f"[{csv_path}] Používám human sloupec: {human_col}")

        judge_counts = prepare_counts(df[judge_col])
        human_counts = prepare_counts(df[human_col])

        if judge_counts.sum() == 0:
            raise ValueError(f"Ve sloupci '{judge_col}' v souboru '{csv_path}' se nepodařilo najít žádná validní skóre 0-10.")

        if human_counts.sum() == 0:
            raise ValueError(f"Ve sloupci '{human_col}' v souboru '{csv_path}' se nepodařilo najít žádná validní skóre 0-10.")

        ax = axes[i]
        ax.bar([j - width / 2 for j in x], judge_counts.values, width=width, label=judge_col)
        ax.bar([j + width / 2 for j in x], human_counts.values, width=width, label=human_col)

        ax.set_xticks(x)
        ax.set_xlabel("Skóre")
        ax.set_ylabel("Četnost")
        ax.set_title(csv_path)
        ax.set_ylim(0, y_max)
        ax.legend()

    for i in range(num_files, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    output_path = "all_graphs.png"
    plt.savefig(output_path, dpi=200)
    print(f"Graf uložen do: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()

# python3 figs.py out_1_harmful_fix_updated.csv out_2_harmful_fix_updated.csv out_3_harmful_fix_updated.csv out_4_harmful_fix_updated.csv out_5_harmful_fix_updated.csv
# python3 figs.py out_2_harmful_fix_updated_simulated.csv out_2_harmful_fix_updated_simulated.csv out_3_harmful_fix_updated_simulated.csv out_4_harmful_fix_updated_simulated.csv out_5_harmful_fix_updated_simulated.csv
# python3 figs.py show_1.csv show_2.csv show_3.csv show_4.csv show_5.csv
