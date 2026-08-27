"""
Grid plot generator and average results calculator script for RecBole thesis evaluation.

Generates:
  1. Unified 2x3 metric grid plots:
     - results/<dataset>_baseline_grid.png (e.g. results/30music_baseline_grid.png)
     - results/<dataset>_modification_grid.png (e.g. results/30music_modification_grid.png)
  2. Averaged metric results (JSON):
     - results/<dataset>_baseline_average_results.json
     - results/<dataset>_modification_average_results.json
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CUSTOM_COLORS = ["#356070", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51", "#7209b7", "#4361ee"]
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

SAVED_DIR = Path("saved")
RESULTS_DIR = Path("results")

_PCOUNT_RE = re.compile(r'pcount\[([^\]]+)\]')


def _extract_pcount(dataset_name: str) -> str:
    """Extract the pcount value from a dataset name like '...pcount[5]_...' or '...pcount[0.1]_...'."""
    m = _PCOUNT_RE.search(dataset_name)
    return m.group(1) if m else "unknown"


FIXED_BASELINE_ORDER = [
    "FPMC",
    "GRU4Rec",
    "NARM",
    "STAMP",
    "CORE",
    "SRGNN",
    "SASRec",
]

MODIFICATION_MODELS = [
    "GRU4Rec",
    "GRU4RecF",
    "GRU4RecF_Rerank",
]

MODEL_DISPLAY_LABELS = {
    "FPMC": "FPMC",
    "GRU4Rec": "GRU4Rec",
    "NARM": "NARM",
    "STAMP": "STAMP",
    "CORE": "CORE",
    "SRGNN": "SRGNN",
    "SASRec": "SASRec",
    "GRU4RecF": "GRU4RecF",
    "GRU4RecF_Rerank": "GRU4RecF+",
}

MODEL_MARKERS = {
    "FPMC": "o",
    "GRU4Rec": "s",
    "NARM": "^",
    "STAMP": "D",
    "CORE": "8",
    "SRGNN": "v",
    "SASRec": "p",
    "GRU4RecF": "s",
    "GRU4RecF_Rerank": "^",
}

METRIC_DISPLAY_NAMES = {
    "precision": "Precision",
    "recall": "Recall",
    "mrr": "MRR",
    "ndcg": "NDCG",
    "itemcoverage": "Item Coverage",
    "averagepopularity": "Average Popularity",
}

KS = [5, 10, 20]

METRIC_BASES = [
    "precision",
    "recall",
    "mrr",
    "ndcg",
    "itemcoverage",
    "averagepopularity",
]


def get_model_order_idx(model_name: str) -> int:
    name_upper = model_name.upper()
    for idx, order_name in enumerate(FIXED_BASELINE_ORDER):
        if order_name.upper() == name_upper:
            return idx
    return 999


def render_metric_grid(
    models: list[str],
    metric_data: dict[str, dict[str, list[float]]],
    ks: list[int],
    output_file: Path,
    nrows: int = 3,
    ncols: int = 2,
    row_height: float = 2.6,
    col_width: float = 3.0,
    legend_height: float = 0.8,
):
    """Renders all METRIC_BASES as a 3x2 grid in a single figure with one shared legend."""
    fig_width = col_width * ncols
    fig_height = row_height * nrows + legend_height

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows + 1, ncols,
        height_ratios=[legend_height] + [row_height] * nrows,
    )

    legend_ax = fig.add_subplot(gs[0, :])
    legend_ax.axis("off")

    axes_flat = [fig.add_subplot(gs[r + 1, c]) for r in range(nrows) for c in range(ncols)]

    palette = CUSTOM_COLORS[:len(models)] if len(models) <= len(CUSTOM_COLORS) else sns.color_palette(CUSTOM_COLORS, n_colors=len(models))
    legend_handles = {}

    for idx, base_m in enumerate(METRIC_BASES):
        ax = axes_flat[idx]
        display_name = METRIC_DISPLAY_NAMES.get(base_m, base_m.upper())
        is_cov = base_m == "itemcoverage"

        curves = metric_data.get(base_m, {})
        for m_idx, model in enumerate(models):
            if model in curves and len(curves[model]) == len(ks):
                vals = curves[model]
                color = palette[m_idx]
                marker = MODEL_MARKERS.get(model, "o")
                line, = ax.plot(
                    ks, vals,
                    marker=marker,
                    color=color,
                    label=MODEL_DISPLAY_LABELS.get(model, model),
                    linewidth=1.6,
                    markersize=5,
                )
                if model not in legend_handles:
                    legend_handles[model] = line

        unit = " (%)" if is_cov else ""
        ax.set_xlabel("K", fontweight="bold", fontsize=10, labelpad=8)
        ax.set_ylabel(f"{display_name}{unit}", fontweight="bold", fontsize=10, labelpad=8)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks], fontweight="bold", fontsize=9.5)
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda val, pos: f"{val:g}".replace(".", ",")))
        plt.setp(ax.get_yticklabels(), fontweight="normal", fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.4, axis="both")
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)

    for idx in range(len(METRIC_BASES), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.08, wspace=0.125, hspace=0.125)

    ordered_handles = [legend_handles[m] for m in models if m in legend_handles]
    ordered_labels = [MODEL_DISPLAY_LABELS.get(m, m) for m in models if m in legend_handles]

    if ordered_handles:
        fig.legend(
            ordered_handles,
            ordered_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=max(1, min(len(ordered_handles), 4)),
            frameon=False,
            fontsize=11,
        )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_baseline_averages(pcount: str, ds_group: str, aggregated: dict):
    """Saves averaged baseline metrics to a clean JSON file in results/."""
    json_data = {}
    for m in sorted(aggregated.keys(), key=get_model_order_idx):
        m_label = MODEL_DISPLAY_LABELS.get(m, m)
        json_data[m_label] = {}
        for base_m in METRIC_BASES:
            for k in KS:
                key = f"{base_m}@{k}".lower()
                if key in aggregated[m]:
                    json_data[m_label][key] = round(aggregated[m][key], 4)

    out_file = RESULTS_DIR / f"{ds_group}_baseline_average_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Saved baseline averages: {out_file}")


def save_modification_averages(pcount: str, ds_group: str, avg_metrics: dict):
    """Saves averaged modification metrics to a clean JSON file in results/."""
    json_data = {}
    for model in MODIFICATION_MODELS:
        m_label = MODEL_DISPLAY_LABELS.get(model, model)
        json_data[m_label] = {}
        for base_m in METRIC_BASES:
            for k in KS:
                key = f"{base_m}@{k}".lower()
                if key in avg_metrics and model in avg_metrics[key]:
                    json_data[m_label][key] = round(avg_metrics[key][model], 4)

    out_file = RESULTS_DIR / f"{ds_group}_modification_average_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Saved modification averages: {out_file}")


def load_raw_model_results() -> dict[tuple, dict[str, dict[str, dict]]]:
    """Returns raw_results[(pcount, ds_group)][model_name][dataset_name] = res."""
    raw_model_results = defaultdict(lambda: defaultdict(dict))
    if not SAVED_DIR.exists():
        return raw_model_results

    for folder in sorted(SAVED_DIR.iterdir()):
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        parts = folder.name.split("_", 1)
        if len(parts) < 2:
            continue
        model_name = parts[0]
        dataset_name = parts[1]
        ds_group = dataset_name.split("__")[0]
        pcount = _extract_pcount(dataset_name)

        res_file = None
        for cand_name in ["results_1.json", "results.json", "result.json", "test_result.json", "test_results.json"]:
            cand_p = folder / cand_name
            if cand_p.exists():
                res_file = cand_p
                break

        if res_file:
            try:
                with open(res_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    res = data.get("test_result", data)
                    if isinstance(res, dict):
                        res = {str(k).lower(): float(v) for k, v in res.items() if isinstance(v, (int, float))}
                    raw_model_results[(pcount, ds_group)][model_name][dataset_name] = res
            except Exception:
                pass

    return raw_model_results


def generate_baseline_grid(pcount: str, ds_group: str, group_raw_results: dict[str, dict[str, dict]]):
    available = [m for m in group_raw_results.keys() if m.lower() != "gru4recf"]
    models = sorted(available, key=get_model_order_idx)
    if not models:
        return

    dataset_sets = [set(group_raw_results[m].keys()) for m in models if m in group_raw_results]
    common_datasets = sorted(list(set.intersection(*dataset_sets))) if dataset_sets else []

    model_splits = defaultdict(list)
    for m in models:
        target_ds = common_datasets if common_datasets else sorted(list(group_raw_results[m].keys()))
        for ds in target_ds:
            if ds in group_raw_results[m]:
                model_splits[m].append(group_raw_results[m][ds])

    aggregated = defaultdict(dict)
    for m in models:
        splits = model_splits[m]
        if not splits:
            continue
        for base_m in METRIC_BASES:
            for k in KS:
                key = f"{base_m}@{k}".lower()
                vals = [s[key] for s in splits if key in s]
                if vals:
                    aggregated[m][key] = float(np.mean(vals))

    # Zapis uśrednionych wyników do pliku JSON
    save_baseline_averages(pcount, ds_group, aggregated)

    metric_data = {}
    for base_m in METRIC_BASES:
        model_curves = defaultdict(list)
        is_cov = base_m == "itemcoverage"
        for m in models:
            for k in KS:
                key = f"{base_m}@{k}".lower()
                if m in aggregated and key in aggregated[m]:
                    v = aggregated[m][key]
                    if is_cov and v <= 1.0:
                        v *= 100
                    model_curves[m].append(v)
        metric_data[base_m] = model_curves

    out_file = RESULTS_DIR / f"{ds_group}_baseline_grid.png"
    render_metric_grid(
        models=models,
        metric_data=metric_data,
        ks=KS,
        output_file=out_file,
    )
    print(f"Generated baseline grid: {out_file}")


def load_modification_data() -> dict[tuple, dict[str, dict[str, dict]]]:
    """Returns combined_splits[(pcount, ds_group)][split_suffix] = split_metrics."""
    combined_splits = defaultdict(dict)
    if not SAVED_DIR.exists():
        return combined_splits

    # Sprawdź również pliki zbiorcze all_rerank w SAVED_DIR jako fallback
    all_rerank_data = {}
    for all_cand in SAVED_DIR.glob("*all_rerank*.json"):
        try:
            with open(all_cand, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, dict):
                    all_rerank_data.update(content)
        except Exception:
            pass

    for folder in sorted(SAVED_DIR.iterdir()):
        if not folder.is_dir() or folder.name.startswith("_") or not folder.name.startswith("GRU4Rec_"):
            continue

        split_suffix = folder.name[len("GRU4Rec_"):]
        ds_group = split_suffix.split("__")[0]
        pcount = _extract_pcount(split_suffix)

        g_res = None
        for cand_name in ["results_1.json", "results.json", "result.json", "test_result.json", "test_results.json"]:
            cand_p = folder / cand_name
            if cand_p.exists():
                g_res = cand_p
                break

        f_dir = SAVED_DIR / f"GRU4RecF_{split_suffix}"
        f_res = None
        if f_dir.exists():
            for cand_name in ["results_1.json", "results.json", "result.json", "test_result.json", "test_results.json"]:
                cand_p = f_dir / cand_name
                if cand_p.exists():
                    f_res = cand_p
                    break

        rerank_file = None
        if f_dir.exists():
            for cand in [
                "rerank_artist_tags_comparison.json",
                "rerank_track_tags_comparison.json",
                "rerank_comparison.json",
                "reranking_comparison.json",
            ]:
                cand_path = f_dir / cand
                if cand_path.exists():
                    rerank_file = cand_path
                    break

        if not g_res:
            continue

        g_data = {}
        f_data = {}
        r_data = {}

        try:
            with open(g_res, "r", encoding="utf-8") as f:
                raw_g = json.load(f)
                res_part = raw_g.get("test_result", raw_g)
                if isinstance(res_part, dict):
                    g_data = {str(k).lower(): float(v) for k, v in res_part.items() if isinstance(v, (int, float))}
        except Exception:
            pass

        if rerank_file:
            try:
                with open(rerank_file, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    base_part = content.get("baseline", {})
                    rerank_part = content.get("reranking", {})
                    f_data = {str(k).lower(): float(v) for k, v in base_part.items() if isinstance(v, (int, float))}
                    r_data = {str(k).lower(): float(v) for k, v in rerank_part.items() if isinstance(v, (int, float))}
            except Exception:
                pass
        elif split_suffix in all_rerank_data:
            content = all_rerank_data[split_suffix]
            base_part = content.get("baseline", {})
            rerank_part = content.get("reranking", {})
            f_data = {str(k).lower(): float(v) for k, v in base_part.items() if isinstance(v, (int, float))}
            r_data = {str(k).lower(): float(v) for k, v in rerank_part.items() if isinstance(v, (int, float))}
        elif f_res:
            try:
                with open(f_res, "r", encoding="utf-8") as f:
                    raw_f = json.load(f)
                    res_part = raw_f.get("test_result", raw_f)
                    f_data = {str(k).lower(): float(v) for k, v in res_part.items() if isinstance(v, (int, float))}
                    r_data = dict(f_data)
            except Exception:
                pass

        if g_data and f_data:
            split_metrics = {}
            all_keys = set(g_data.keys()).union(f_data.keys()).union(r_data.keys())
            for k in all_keys:
                split_metrics[k.lower()] = {
                    "GRU4Rec": float(g_data.get(k, 0.0)),
                    "GRU4RecF": float(f_data.get(k, 0.0)),
                    "GRU4RecF_Rerank": float(r_data.get(k, f_data.get(k, 0.0))),
                }
            combined_splits[(pcount, ds_group)][split_suffix] = split_metrics

    return combined_splits


def generate_modification_grid(pcount: str, ds_group: str, splits_data: dict[str, dict[str, dict]]):
    if not splits_data:
        return

    splits = list(splits_data.keys())
    avg_metrics = defaultdict(dict)
    for model in MODIFICATION_MODELS:
        for base_m in METRIC_BASES:
            for k in KS:
                key = f"{base_m}@{k}".lower()
                vals = [splits_data[s][key][model] for s in splits if key in splits_data[s] and model in splits_data[s][key]]
                if vals:
                    avg_metrics[key][model] = float(np.mean(vals))

    # Zapis uśrednionych wyników do pliku JSON
    save_modification_averages(pcount, ds_group, avg_metrics)

    metric_data = {}
    for base_m in METRIC_BASES:
        model_curves = defaultdict(list)
        is_cov = base_m == "itemcoverage"
        for model in MODIFICATION_MODELS:
            for k in KS:
                key = f"{base_m}@{k}".lower()
                if key in avg_metrics and model in avg_metrics[key]:
                    v = avg_metrics[key][model]
                    if is_cov and v <= 1.0:
                        v *= 100
                    model_curves[model].append(v)
        metric_data[base_m] = model_curves

    out_file = RESULTS_DIR / f"{ds_group}_modification_grid.png"
    render_metric_grid(
        models=MODIFICATION_MODELS,
        metric_data=metric_data,
        ks=KS,
        output_file=out_file,
    )
    print(f"Generated modification grid: {out_file}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_results = load_raw_model_results()
    mod_splits = load_modification_data()

    all_keys = sorted(list(set(raw_results.keys()).union(mod_splits.keys())))

    for pcount, ds_group in all_keys:
        print(f"Processing pcount={pcount}, ds_group={ds_group}")
        if (pcount, ds_group) in raw_results:
            generate_baseline_grid(pcount, ds_group, raw_results[(pcount, ds_group)])
        if (pcount, ds_group) in mod_splits:
            generate_modification_grid(pcount, ds_group, mod_splits[(pcount, ds_group)])


if __name__ == "__main__":
    main()