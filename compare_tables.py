import json
import re
from pathlib import Path

RESULTS_PATH = Path("results")

METRIC_ORDER = ["precision", "recall", "mrr", "ndcg", "itemcoverage", "averagepopularity"]

def parse_latex_row(line):
    clean = line.strip().rstrip("\\\\").strip()
    clean = re.sub(r"\\textbf\{([^}]*)\}", r"\1", clean)
    clean = re.sub(r"\\underline\{([^}]*)\}", r"\1", clean)

    parts = [p.strip() for p in clean.split("&")]
    if len(parts) != 7:
        return None, None

    model = parts[0].strip()
    if model == "SR-GNN":
        model = "SRGNN"

    values = []
    for v in parts[1:]:
        v = v.replace(",", ".").strip()
        try:
            values.append(float(v))
        except ValueError:
            return None, None

    return model, values

def compare(dataset, suffix):
    json_path = RESULTS_PATH / f"{dataset}_{suffix}_average_results.json"
    tex_path = RESULTS_PATH / dataset / f"{suffix}.tex"

    with open(json_path, "r") as f:
        data = json.load(f)

    with open(tex_path, "r") as f:
        tex_lines = f.readlines()

    print(f"\n{'='*70}")
    print(f"  {dataset} / {suffix}")
    print(f"{'='*70}")

    current_k = None
    mismatches = 0
    matches = 0

    for line in tex_lines:
        line = line.strip()

        k_match = re.search(r"K\s*=\s*(\d+)", line)
        if k_match:
            current_k = int(k_match.group(1))
            continue

        if current_k is None:
            continue

        model, tex_values = parse_latex_row(line)
        if model is None:
            continue

        if model not in data:
            print(f"  WARNING: {model} not found in JSON!")
            continue

        for i, metric in enumerate(METRIC_ORDER):
            key = f"{metric}@{current_k}"
            json_val = data[model].get(key)
            tex_val = tex_values[i]

            if json_val is None:
                print(f"  WARNING: {model} {key} not in JSON")
                continue

            if metric == "averagepopularity":
                json_rounded = round(json_val, 2)
                match = abs(json_rounded - tex_val) < 0.015
            else:
                json_rounded = round(json_val, 4)
                match = abs(json_rounded - tex_val) < 0.00005

            if match:
                matches += 1
            else:
                mismatches += 1
                print(f"  MISMATCH: {model} {key}: JSON={json_val} -> {json_rounded} vs LaTeX={tex_val}")

    print(f"\n  Total: {matches} matches, {mismatches} mismatches")
    return mismatches

total = 0
for ds in ["30music", "lastfm1k"]:
    total += compare(ds, "baseline")
    total += compare(ds, "modification")

print(f"\n{'='*70}")
print(f"  ALL MATCH!" if total == 0 else f"  TOTAL MISMATCHES: {total}")
print(f"{'='*70}")
