#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

OUT_ZIP="${1:-submission_prompt_attacker.zip}"
MANIFEST="submission_manifest.txt"

rm -f "$OUT_ZIP" "$MANIFEST"

touch "$MANIFEST"

add_file() {
    local file="$1"
    if [[ -f "$file" ]]; then
        printf '%s\n' "$file" >> "$MANIFEST"
    fi
}

add_tree() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        find "$dir" -type f \
            ! -path '*/.git/*' \
            ! -path '*/__pycache__/*' \
            ! -path '*/jobs/*' \
            ! -name '*.pyc' \
            ! -name '*.log' \
            ! -name '.DS_Store' \
            | sort >> "$MANIFEST"
    fi
}

# Root project metadata.
add_file "README.md"
add_file "LICENSE"
add_file "requirements.txt"
add_file ".gitignore"

# Main project sources and configuration.
add_file "prompt_attacker/run_orchestrator.py"
add_file "prompt_attacker/config_orchestrator.yaml"
add_file "prompt_attacker/oponent_tutorial_steps.txt"

add_tree "prompt_attacker/attacks"
add_tree "prompt_attacker/defense"
# Evaluation source code only. Generated plots/tables under evaluate/ are not
# needed for reproducing the workflows and can make the submission archive noisy.
if [[ -d "prompt_attacker/evaluate" ]]; then
    find "prompt_attacker/evaluate" -maxdepth 1 -type f -name '*.py' | sort >> "$MANIFEST"
fi
add_tree "prompt_attacker/scripts"

# Small reproducible example inputs.
add_tree "prompt_attacker/dataset/oponent_show"
add_file "prompt_attacker/evaluate/selected_examples.json"

# Trained example defense rule.
add_file "prompt_attacker/defense/defense_rule_orchestrator.json"

# Small example results: top-level oponent_show JSONs and selected eval outputs.
if [[ -d "prompt_attacker/results/oponent_show" ]]; then
    find "prompt_attacker/results/oponent_show" -maxdepth 1 -type f \
        \( -name '*.json' -o -name '*.csv' \) \
        | sort >> "$MANIFEST"
fi
add_file "prompt_attacker/results/oponent_show/eval/NO_DEFENSE/falcon3:3b.csv"
add_file "prompt_attacker/results/oponent_show/eval/NO_DEFENSE/falcon3:3b.summary.json"

# Small helper scripts only, not generated CSV artifacts.
if [[ -d "prompt_attacker/supporting_scripts" ]]; then
    find "prompt_attacker/supporting_scripts" -type f -name '*.py' \
        ! -path '*/__pycache__/*' \
        | sort >> "$MANIFEST"
fi

# Deduplicate and remove anything that should never enter the submission zip.
sort -u "$MANIFEST" -o "$MANIFEST"

grep -vE '(^|/)models(/|$)' "$MANIFEST" > "$MANIFEST.tmp"
mv "$MANIFEST.tmp" "$MANIFEST"

grep -vE '(^|/)results/(harmful|benign)(/|$)' "$MANIFEST" > "$MANIFEST.tmp"
mv "$MANIFEST.tmp" "$MANIFEST"

grep -vE '(^|/)(ollama|ollama_test)(/|$)' "$MANIFEST" > "$MANIFEST.tmp"
mv "$MANIFEST.tmp" "$MANIFEST"

grep -vE '(^|/)jobs(/|$)' "$MANIFEST" > "$MANIFEST.tmp"
mv "$MANIFEST.tmp" "$MANIFEST"

if grep -E '(^|/)models(/|$)|(^|/)jobs(/|$)|__pycache__|\.pyc$|\.log$' "$MANIFEST"; then
    echo "Refusing to create zip: manifest still contains excluded files." >&2
    exit 1
fi

zip -q -@ "$OUT_ZIP" < "$MANIFEST"

printf 'Created %s\n' "$OUT_ZIP"
printf 'Manifest: %s\n' "$MANIFEST"
printf 'Files: '
wc -l < "$MANIFEST"
printf 'Archive size: '
du -h "$OUT_ZIP" | awk '{print $1}'
