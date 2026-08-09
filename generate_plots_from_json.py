"""
Script for generating visualization plots directly from saved `average_results.json` files.

Generates:
  1. Single Bar Plots per Metric @ K (5, 10, 20)
  2. Grouped Bar Plots comparing K=5 and K=10 side-by-side per model
  3. Line Curves (Metric vs K=5, 10, 20) per model

Works for both Baseline models and GRU4Rec Modifications.
"""

import os
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Visual style configuration matching thesis aesthetic
CUSTOM_COLORS = ["#356070", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]
sns.set_palette(CUSTOM_COLORS)
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral", "serif"],
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

RESULTS_DIR = Path("results")

FIXED_BASELINE_ORDER = [
    "FPMC",
    "GRU4Rec",
    "NARM",
    "STAMP",
    "SRGNN",
    "SASRec",
]

MODIFICATION_MODELS = [
    "GRU4Rec",
    "GRU4RecF",
    "GRU4RecF+",
    "GRU4RecF_Rerank",
]

MODEL_DISPLAY_LABELS = {
    "FPMC": "FPMC",
    "GRU4Rec": "GRU4Rec",
    "NARM": "NARM",
    "STAMP": "STAMP",
    "SRGNN": "SRGNN",
    "SASRec": "SASRec",
    "GRU4RecF": "GRU4RecF",
    "GRU4RecF+": "GRU4RecF+",
    "GRU4RecF_Rerank": "GRU4RecF+",
}

MODEL_MARKERS = {
    "FPMC": "o",
    "GRU4Rec": "s",
    "NARM": "^",
    "STAMP": "D",
    "SRGNN": "v",
    "SASRec": "p",
    "GRU4RecF": "s",
    "GRU4RecF+": "^",
    "GRU4RecF_Rerank": "^",
}

METRIC_DISPLAY_NAMES = {
    "recall": "Recall",
    "precision": "Precision",
    "mrr": "MRR",
    "ndcg": "NDCG",
    "averagepopularity": "Average Popularity",
    "itemcoverage": "Item Coverage",
}

METRIC_BASES = [
    "recall",
    "precision",
    "mrr",
    "ndcg",
    "averagepopularity",
    "itemcoverage",
]

KS = [5, 10, 20]


def get_model_order_idx(model_name: str) -> int:
    name_upper = model_name.upper()
    for idx, order_name in enumerate(FIXED_BASELINE_ORDER):
        if order_name.upper() == name_upper:
            return idx
    return 999


def render_single_bar_chart(
    display_labels: list[str],
    vals: list[float],
    y_label: str,
    output_file: Path,
    is_coverage: bool = False,
    is_pop: bool = False,
    red_line_value: float | None = None,
):
    """Renders a single bar chart for a metric at a specific K."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x_pos = np.arange(len(display_labels))
    palette = CUSTOM_COLORS[:len(display_labels)] if len(display_labels) <= len(CUSTOM_COLORS) else sns.color_palette(CUSTOM_COLORS, n_colors=len(display_labels))

    rects = ax.bar(
        x_pos,
        vals,
        color=palette,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.6,
        width=0.65,
    )

    if red_line_value is not None:
        ax.axhline(y=red_line_value, color="red", linestyle="--", alpha=0.5, linewidth=0.9, zorder=1)

    for rect in rects:
        height = rect.get_height()
        if is_pop:
            fmt_val = f"{height:.1f}"
        elif is_coverage:
            fmt_val = f"{height:.2f}%" if height > 1.0 else f"{height * 100:.2f}%"
        else:
            fmt_val = f"{height:.4f}"

        fmt_val = fmt_val.replace(".", ",")
        ax.annotate(
            fmt_val,
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Model", fontweight="bold", fontsize=11, labelpad=12)
    ax.set_ylabel(y_label, fontweight="bold", fontsize=11, labelpad=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_labels, fontweight="bold", fontsize=10.5)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda val, pos: f"{val:g}".replace(".", ",")))
    plt.setp(ax.get_yticklabels(), fontweight="normal")

    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.set_axisbelow(True)

    max_y = max(vals) if vals else 1.0
    ax.set_ylim(0, max_y * 1.14)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)
    plt.close(fig)


def render_grouped_bar_chart_k5_k10(
    display_labels: list[str],
    vals_k5: list[float],
    vals_k10: list[float],
    y_label: str,
    output_file: Path,
    is_coverage: bool = False,
    is_pop: bool = False,
):
    """Renders a grouped bar chart comparing K=5 and K=10 side-by-side per model."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(display_labels))
    width = 0.35

    color_k5 = "#356070"   # Dark teal for K=5
    color_k10 = "#8ab17d"  # Sage green for K=10

    rects1 = ax.bar(
        x - width / 2,
        vals_k5,
        width,
        label="K = 5",
        color=color_k5,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.6,
    )
    rects2 = ax.bar(
        x + width / 2,
        vals_k10,
        width,
        label="K = 10",
        color=color_k10,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.6,
    )

    for rects, vals in [(rects1, vals_k5), (rects2, vals_k10)]:
        for rect in rects:
            height = rect.get_height()
            if is_pop:
                fmt_val = f"{height:.1f}"
            elif is_coverage:
                fmt_val = f"{height:.2f}%" if height > 1.0 else f"{height * 100:.2f}%"
            else:
                fmt_val = f"{height:.4f}"
            fmt_val = fmt_val.replace(".", ",")
            ax.annotate(
                fmt_val,
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    ax.set_xlabel("Model", fontweight="bold", fontsize=11, labelpad=10)
    ax.set_ylabel(y_label, fontweight="bold", fontsize=11, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontweight="bold", fontsize=10.5)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda val, pos: f"{val:g}".replace(".", ",")))
    plt.setp(ax.get_yticklabels(), fontweight="normal")

    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="lightgray", fontsize=9.5)

    max_y = max(max(vals_k5), max(vals_k10)) if vals_k5 and vals_k10 else 1.0
    ax.set_ylim(0, max_y * 1.16)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)
    plt.close(fig)


def render_line_chart(
    models: list[str],
    model_curves: dict[str, list[float]],
    ks: list[int],
    y_label: str,
    output_file: Path,
):
    """Renders line chart showing Metric vs K curves for models."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    palette = CUSTOM_COLORS[:len(models)] if len(models) <= len(CUSTOM_COLORS) else sns.color_palette(CUSTOM_COLORS, n_colors=len(models))

    for idx, model in enumerate(models):
        if model in model_curves and len(model_curves[model]) == len(ks):
            vals = model_curves[model]
            marker = MODEL_MARKERS.get(model, "o")
            color = palette[idx]
            ax.plot(
                ks,
                vals,
                marker=marker,
                color=color,
                label=MODEL_DISPLAY_LABELS.get(model, model),
                linewidth=2.0,
                markersize=5,
            )

    ax.set_xlabel("K", fontweight="bold", fontsize=11, labelpad=12)
    ax.set_ylabel(y_label, fontweight="bold", fontsize=11, labelpad=12)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], fontweight="bold", fontsize=10.5)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda val, pos: f"{val:g}".replace(".", ",")))
    plt.setp(ax.get_yticklabels(), fontweight="normal")

    ax.grid(True, linestyle="--", alpha=0.4, axis="both")
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=True, facecolor="white", edgecolor="lightgray", fontsize=9.5)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)
    plt.close(fig)


def process_average_results_json(json_path: Path):
    """Reads an `average_results.json` file and generates all 3 types of plots."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return

    if not data:
        print(f"File {json_path} is empty. Skipping.")
        return

    # Parent directory is baseline or modification
    group_dir = json_path.parent
    bar_dir = group_dir / "bar"
    grouped_bar_dir = group_dir / "grouped_bar_k5_k10"
    line_dir = group_dir / "line"

    # Identify models available in data
    raw_models = list(data.keys())

    # Check if baseline or modification
    is_modification = "modification" in str(json_path)

    if is_modification:
        models = [m for m in MODIFICATION_MODELS if m in raw_models or MODEL_DISPLAY_LABELS.get(m, m) in raw_models]
        if not models:
            models = raw_models
    else:
        models = sorted([m for m in raw_models if m.lower() != "gru4recf"], key=get_model_order_idx)

    if not models:
        print(f"No valid models found in {json_path}. Skipping.")
        return

    print(f"Processing {json_path}: models = {models}")

    # Helper function to extract value for model and metric_key
    def get_val(model: str, key: str) -> float | None:
        m_dict = data.get(model, data.get(MODEL_DISPLAY_LABELS.get(model, model), {}))
        if isinstance(m_dict, dict):
            val = m_dict.get(key.lower(), m_dict.get(key, None))
            if isinstance(val, dict):
                val = val.get("mean", None)
            return val
        return None

    # 1. Single Bar Plots per metric base and K
    for base_m in METRIC_BASES:
        display_name = METRIC_DISPLAY_NAMES.get(base_m, base_m.upper())
        is_cov = base_m == "itemcoverage"
        is_pop = base_m == "averagepopularity"

        for k in KS:
            key = f"{base_m}@{k}".lower()
            vals = []
            display_labels = []

            for m in models:
                v = get_val(m, key)
                if v is not None:
                    if is_cov and v <= 1.0:
                        v *= 100
                    vals.append(v)
                    display_labels.append(MODEL_DISPLAY_LABELS.get(m, m))

            if not vals:
                continue

            unit = " (%)" if is_cov else ""
            y_label = f"{display_name} @ {k}{unit}"
            out_file = bar_dir / f"{base_m}@{k}.png"

            red_line_val = None
            if is_modification and "GRU4Rec" in display_labels:
                gru_idx = display_labels.index("GRU4Rec")
                red_line_val = vals[gru_idx]

            render_single_bar_chart(
                display_labels=display_labels,
                vals=vals,
                y_label=y_label,
                output_file=out_file,
                is_coverage=is_cov,
                is_pop=is_pop,
                red_line_value=red_line_val,
            )

    # 2. Grouped Bar Plots for K=5 and K=10
    for base_m in METRIC_BASES:
        display_name = METRIC_DISPLAY_NAMES.get(base_m, base_m.upper())
        is_cov = base_m == "itemcoverage"
        is_pop = base_m == "averagepopularity"

        key_k5 = f"{base_m}@5".lower()
        key_k10 = f"{base_m}@10".lower()

        vals_k5 = []
        vals_k10 = []
        display_labels = []

        for m in models:
            v5 = get_val(m, key_k5)
            v10 = get_val(m, key_k10)

            if v5 is not None and v10 is not None:
                if is_cov:
                    if v5 <= 1.0: v5 *= 100
                    if v10 <= 1.0: v10 *= 100
                vals_k5.append(v5)
                vals_k10.append(v10)
                display_labels.append(MODEL_DISPLAY_LABELS.get(m, m))

        if not vals_k5 or not vals_k10:
            continue

        unit = " (%)" if is_cov else ""
        y_label = f"{display_name}{unit}"
        out_file = grouped_bar_dir / f"{base_m}_k5_k10.png"

        render_grouped_bar_chart_k5_k10(
            display_labels=display_labels,
            vals_k5=vals_k5,
            vals_k10=vals_k10,
            y_label=y_label,
            output_file=out_file,
            is_coverage=is_cov,
            is_pop=is_pop,
        )

    # 3. Line Charts (Metric vs K)
    for base_m in METRIC_BASES:
        display_name = METRIC_DISPLAY_NAMES.get(base_m, base_m.upper())
        is_cov = base_m == "itemcoverage"

        model_curves = defaultdict(list)
        for m in models:
            for k in KS:
                key = f"{base_m}@{k}".lower()
                v = get_val(m, key)
                if v is not None:
                    if is_cov and v <= 1.0:
                        v *= 100
                    model_curves[m].append(v)

        if not model_curves:
            continue

        unit = " (%)" if is_cov else ""
        y_label = f"{display_name}{unit}"
        out_line_file = line_dir / f"{base_m}.png"

        render_line_chart(
            models=models,
            model_curves=model_curves,
            ks=KS,
            y_label=y_label,
            output_file=out_line_file,
        )


def main():
    """Finds all `average_results.json` files in `results/` and generates plots."""
    if not RESULTS_DIR.exists():
        print(f"Directory '{RESULTS_DIR}' does not exist.")
        return

    json_files = list(RESULTS_DIR.rglob("average_results.json"))
    if not json_files:
        print(f"No `average_results.json` files found in '{RESULTS_DIR}'.")
        return

    print(f"Found {len(json_files)} `average_results.json` file(s):")
    for jf in json_files:
        print(f" - {jf}")

    for jf in json_files:
        process_average_results_json(jf)

    print("\n✓ Plot generation completed successfully!")


if __name__ == "__main__":
    main()
