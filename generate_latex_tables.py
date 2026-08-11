"""
Script to generate LaTeX tables from evaluation results for Last.fm-1k and 30Music datasets.

Generates 4 LaTeX tables:
1. Baseline models on Last.fm-1k
2. GRU4Rec modifications on Last.fm-1k
3. Baseline models on 30Music
4. GRU4Rec modifications on 30Music
"""

import json
import os
from pathlib import Path

RESULTS_DIR = Path("results/5")
OUTPUT_TEX_FILE = Path("results/tables.tex")

BASE_MODELS = ["FPMC", "GRU4Rec", "NARM", "STAMP", "SRGNN", "SASRec"]
MOD_MODELS = ["GRU4Rec", "GRU4RecF", "GRU4RecF+"]
KS = [5, 10, 20]


def load_results_json(pcount_dir: Path, dataset: str, mode: str) -> dict:
    """Load average_results.json for a given dataset ('lastfm1k' or '30music') and mode ('baseline' or 'modification')."""
    json_path = pcount_dir / dataset / mode / "average_results.json"
    if not json_path.exists():
        print(f"Warning: {json_path} not found.")
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_val(metric_name: str, val: float) -> str:
    """Format float metric values: 2 decimal places for averagepopularity, 4 decimal places for others."""
    if "averagepopularity" in metric_name or metric_name.upper() == "POP":
        return f"{val:.2f}"
    else:
        return f"{val:.4f}"


def build_latex_table(data: dict, models: list[str], caption: str, label: str) -> str:
    """Build LaTeX tabularx string for a given result dictionary and list of models."""
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{1.7cm} *{6}{>{\centering\arraybackslash}X}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{P@K} & \textbf{R@K} & \textbf{MRR@K} & \textbf{NDCG@K} & \textbf{COV@K} & \textbf{POP@K} \\")

    for k in KS:
        lines.append(r"\midrule")
        lines.append(f"\\multicolumn{{7}}{{c}}{{\\textbf{{$K = {k}$}}}} \\\\")
        lines.append(r"\midrule")
        for m in models:
            if m not in data:
                # Fallback if model missing
                lines.append(f"{m:<9} & -- & -- & -- & -- & -- & -- \\\\")
                continue
            m_data = data[m]
            p = fmt_val("precision", m_data.get(f"precision@{k}", 0.0))
            r = fmt_val("recall", m_data.get(f"recall@{k}", 0.0))
            mrr = fmt_val("mrr", m_data.get(f"mrr@{k}", 0.0))
            ndcg = fmt_val("ndcg", m_data.get(f"ndcg@{k}", 0.0))
            cov = fmt_val("itemcoverage", m_data.get(f"itemcoverage@{k}", 0.0))
            pop = fmt_val("averagepopularity", m_data.get(f"averagepopularity@{k}", 0.0))

            lines.append(f"{m:<9} & {p} & {r} & {mrr} & {ndcg} & {cov} & {pop} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    lastfm_base = load_results_json(RESULTS_DIR, "lastfm1k", "baseline")
    lastfm_mod = load_results_json(RESULTS_DIR, "lastfm1k", "modification")
    music_base = load_results_json(RESULTS_DIR, "30music", "baseline")
    music_mod = load_results_json(RESULTS_DIR, "30music", "modification")

    tables = [
        build_latex_table(
            lastfm_base,
            BASE_MODELS,
            r"Wyniki modeli bazowych dla zbioru Last.fm-1k przy $K \in \{5, 10, 20\}$",
            "tab:baseline_lastfm1k_all_k",
        ),
        build_latex_table(
            lastfm_mod,
            MOD_MODELS,
            r"Wyniki modyfikacji modelu GRU4Rec dla zbioru Last.fm-1k przy $K \in \{5, 10, 20\}$",
            "tab:modification_lastfm1k_all_k",
        ),
        build_latex_table(
            music_base,
            BASE_MODELS,
            r"Wyniki modeli bazowych dla zbioru 30Music przy $K \in \{5, 10, 20\}$",
            "tab:baseline_30music_all_k",
        ),
        build_latex_table(
            music_mod,
            MOD_MODELS,
            r"Wyniki modyfikacji modelu GRU4Rec dla zbioru 30Music przy $K \in \{5, 10, 20\}$",
            "tab:modification_30music_all_k",
        ),
    ]

    full_output = "\n\n".join(tables) + "\n"

    # Save to LaTeX output file
    OUTPUT_TEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TEX_FILE, "w", encoding="utf-8") as f:
        f.write(full_output)

    print(f"LaTeX tables successfully saved to: {OUTPUT_TEX_FILE}\n")
    print(full_output)


if __name__ == "__main__":
    main()
