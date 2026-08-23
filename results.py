"""
Unified plot generator script for RecBole thesis evaluation.

Generates both bar plots and line plots (metric vs Top-K curves) organized per dataset group:
  - Baseline models       -> results/<dataset>/_baseline/bar/ and results/<dataset>/_baseline/line/
  - GRU4Rec modifications -> results/<dataset>/_modification/bar/ and results/<dataset>/_modification/line/
"""

import json
import math
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import re

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

SAVED_DIR = Path("saved")
RESULTS_DIR = Path("results")
LOG_DIR = Path("log")

_PCOUNT_RE = re.compile(r'pcount\[(\d+)\]')


def _extract_pcount(dataset_name: str) -> str:
    """Extract the pcount value from a dataset name like '...pcount[25]_...'."""
    m = _PCOUNT_RE.search(dataset_name)
    return m.group(1) if m else "unknown"


def get_baseline_dir(pcount: str, ds_group: str) -> Path:
    return RESULTS_DIR / pcount / ds_group / "baseline"


def get_baseline_bar_dir(pcount: str, ds_group: str) -> Path:
    return get_baseline_dir(pcount, ds_group) / "bar"


def get_baseline_line_dir(pcount: str, ds_group: str) -> Path:
    return get_baseline_dir(pcount, ds_group) / "line"


def get_modification_dir(pcount: str, ds_group: str) -> Path:
    return RESULTS_DIR / pcount / ds_group / "modification"


def get_modification_bar_dir(pcount: str, ds_group: str) -> Path:
    return get_modification_dir(pcount, ds_group) / "bar"


def get_modification_line_dir(pcount: str, ds_group: str) -> Path:
    return get_modification_dir(pcount, ds_group) / "line"


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
    "recall": "Recall",
    "ndcg": "NDCG",
    "mrr": "MRR",
    "map": "MAP",
    "precision": "Precision",
    "hit": "Hit Rate",
    "hitrate": "Hit Rate",
    "itemcoverage": "Item Coverage",
    "giniindex": "Gini Index",
    "averagepopularity": "Average Popularity",
}

KS = [5, 10, 20]
ALLOWED_KS = [5, 10, 20]

BASE_ORDER = [
    "recall",
    "precision",
    "mrr",
    "ndcg",
    "averagepopularity",
    "itemcoverage",
]

METRIC_BASES = [
    "recall",
    "ndcg",
    "mrr",
    "precision",
    # "map",
    # "hit",
    "itemcoverage",
    # "giniindex",
    "averagepopularity",
]


def get_metric_sort_key(metric_name: str) -> tuple:
    """Sort key to order metrics logically by base metric and numerically by K (5, 10, 20)."""
    metric_str = metric_name.lower()
    if "@" in metric_str:
        base, k_str = metric_str.split("@", 1)
        try:
            k_val = int(k_str)
        except ValueError:
            k_val = 999
    else:
        base = metric_str
        k_val = 0

    try:
        base_idx = BASE_ORDER.index(base)
    except ValueError:
        base_idx = 999

    return (base_idx, k_val, metric_str)


def get_model_order_idx(model_name: str) -> int:
    name_upper = model_name.upper()
    for idx, order_name in enumerate(FIXED_BASELINE_ORDER):
        if order_name.upper() == name_upper:
            return idx
    return 999


def render_bar_chart(
    display_labels: list[str],
    vals: list[float],
    stds: list[float],
    y_label: str,
    output_file: Path,
    is_coverage: bool = False,
    is_pop: bool = False,
    red_line_value: float | None = None,
):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x_pos = np.arange(len(display_labels))
    palette = CUSTOM_COLORS[:len(display_labels)] if len(display_labels) <= len(CUSTOM_COLORS) else sns.color_palette(CUSTOM_COLORS, n_colors=len(display_labels))

    rects = ax.bar(
        x_pos,
        vals,
        yerr=stds,
        capsize=4,
        color=palette,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.6,
        width=0.65,
    )

    if red_line_value is not None:
        ax.axhline(y=red_line_value, color="red", linestyle="--", alpha=0.5, linewidth=0.9, zorder=1)

    for rect, val, std in zip(rects, vals, stds):
        height = rect.get_height()
        if is_pop:
            fmt_val = f"{height:.1f}"
        elif is_coverage:
            fmt_val = f"{height:.2f}%"
        else:
            fmt_val = f"{height:.4f}"

        fmt_val = fmt_val.replace(".", ",")
        ax.annotate(
            fmt_val,
            xy=(rect.get_x() + rect.get_width() / 2, height + std),
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

    # Enable background horizontal grid lines behind bars
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.set_axisbelow(True)

    max_y = max([v + s for v, s in zip(vals, stds)]) if vals else 1.0
    ax.set_ylim(0, max_y * 1.14)

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
                marker="o",
                color=color,
                label=MODEL_DISPLAY_LABELS.get(model, model),
                linewidth=2.0,
                markersize=4,
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

        res_file = folder / "results_1.json"
        if res_file.exists():
            try:
                with open(res_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    res = data.get("test_result", data)
                    raw_model_results[(pcount, ds_group)][model_name][dataset_name] = res
            except Exception:
                pass

    return raw_model_results


def generate_baseline_plots(pcount: str, ds_group: str, group_raw_results: dict[str, dict[str, dict]]):
    available = [m for m in group_raw_results.keys() if m.lower() != "gru4recf"]
    models = sorted(available, key=get_model_order_idx)
    if not models:
        return

    dataset_sets = [set(group_raw_results[m].keys()) for m in models if m in group_raw_results]
    common_datasets = sorted(list(set.intersection(*dataset_sets))) if dataset_sets else []

    model_splits = defaultdict(list)
    all_metrics = set()
    for m in models:
        target_ds = common_datasets if common_datasets else sorted(list(group_raw_results[m].keys()))
        for ds in target_ds:
            if ds in group_raw_results[m]:
                model_splits[m].append(group_raw_results[m][ds])
                all_metrics.update(group_raw_results[m][ds].keys())

    aggregated = {}
    valid_metrics = [
        m for m in all_metrics
        if "@" in m and m.split("@")[0].lower() in BASE_ORDER and m.split("@")[1].isdigit() and int(m.split("@")[1]) in ALLOWED_KS
    ]
    sorted_metrics = sorted(valid_metrics, key=get_metric_sort_key)

    for m in models:
        splits = model_splits[m]
        if not splits:
            continue
        m_agg = {}
        for metric in sorted_metrics:
            vals = [s[metric] for s in splits if metric in s]
            if vals:
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                m_agg[metric.lower()] = {"mean": mean_val, "std": std_val}
        aggregated[m] = m_agg

    # Build flat json structure for average_results.json: "metric@k": float
    json_aggregated = {}
    for m in models:
        if m in aggregated:
            m_json = {}
            for metric in sorted_metrics:
                metric_key = metric.lower()
                if metric_key in aggregated[m]:
                    m_json[metric_key] = round(aggregated[m][metric_key]["mean"], 4)
            json_aggregated[m] = m_json

    baseline_dir = get_baseline_dir(pcount, ds_group)
    baseline_bar_dir = get_baseline_bar_dir(pcount, ds_group)
    baseline_line_dir = get_baseline_line_dir(pcount, ds_group)

    baseline_dir.mkdir(parents=True, exist_ok=True)
    out_json = baseline_dir / "average_results.json"
    try:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(json_aggregated, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    # 1. Bar Plots
    for base_m in METRIC_BASES:
        display_name = METRIC_DISPLAY_NAMES.get(base_m, base_m.upper())
        file_m_base = "hitrate" if base_m == "hit" else base_m
        is_cov = base_m == "itemcoverage"
        is_pop = base_m == "averagepopularity"

        for k in KS:
            key = f"{base_m}@{k}".lower()
            vals = []
            stds = []
            display_labels = []

            for m in models:
                if m in aggregated and key in aggregated[m]:
                    v = aggregated[m][key]["mean"]
                    s = aggregated[m][key]["std"]
                    if is_cov and v <= 1.0:
                        v *= 100
                        s *= 100
                    vals.append(v)
                    stds.append(s)
                    display_labels.append(MODEL_DISPLAY_LABELS.get(m, m))

            if not vals:
                continue

            unit = " (%)" if is_cov else ""
            y_label = f"{display_name} @ {k}{unit}"
            out_file = baseline_bar_dir / f"{file_m_base}@{k}.png"

            render_bar_chart(
                display_labels=display_labels,
                vals=vals,
                stds=stds,
                y_label=y_label,
                output_file=out_file,
                is_coverage=is_cov,
                is_pop=is_pop,
            )

    # 2. Line Plots
    for base_m in METRIC_BASES:
        display_name = METRIC_DISPLAY_NAMES.get(base_m, base_m.upper())
        file_m_base = "hitrate" if base_m == "hit" else base_m
        is_cov = base_m == "itemcoverage"

        model_curves = defaultdict(list)
        for m in models:
            for k in KS:
                key = f"{base_m}@{k}".lower()
                if m in aggregated and key in aggregated[m]:
                    v = aggregated[m][key]["mean"]
                    if is_cov and v <= 1.0:
                        v *= 100
                    model_curves[m].append(v)

        if not model_curves:
            continue

        unit = " (%)" if is_cov else ""
        y_label = f"{display_name}{unit}"
        out_line_file = baseline_line_dir / f"{file_m_base}.png"

        render_line_chart(
            models=models,
            model_curves=model_curves,
            ks=KS,
            y_label=y_label,
            output_file=out_line_file,
        )


def load_modification_data() -> dict[tuple, dict[str, dict[str, dict]]]:
    """Returns combined_splits[(pcount, ds_group)][split_suffix] = split_metrics."""
    combined_splits = defaultdict(dict)
    if not SAVED_DIR.exists():
        return combined_splits

    for folder in sorted(SAVED_DIR.iterdir()):
        if not folder.is_dir() or folder.name.startswith("_") or not folder.name.startswith("GRU4Rec_"):
            continue

        split_suffix = folder.name[len("GRU4Rec_"):]
        ds_group = split_suffix.split("__")[0]
        pcount = _extract_pcount(split_suffix)

        g_res = folder / "results_1.json"
        f_dir = SAVED_DIR / f"GRU4RecF_{split_suffix}"
        f_res = f_dir / "results_1.json"

        rerank_file = None
        for cand in [
            "rerank_artist_tags_comparison.json",
            "rerank_track_tags_comparison.json",
            "rerank_comparison.json",
        ]:
            cand_path = f_dir / cand
            if cand_path.exists():
                rerank_file = cand_path
                break

        if not g_res.exists() or not f_dir.exists():
            continue

        g_data = {}
        f_data = {}
        r_data = {}

        try:
            with open(g_res, "r", encoding="utf-8") as f:
                g_data = json.load(f).get("test_result", {})
        except Exception:
            pass

        if rerank_file:
            try:
                with open(rerank_file, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    f_data = content.get("baseline", {})
                    r_data = content.get("reranking", {})
            except Exception:
                pass
        elif f_res.exists():
            try:
                with open(f_res, "r", encoding="utf-8") as f:
                    f_data = json.load(f).get("test_result", {})
                    r_data = f_data
            except Exception:
                pass

        if g_data and f_data:
            split_metrics = {}
            all_keys = set(g_data.keys()).union(f_data.keys())
            for k in all_keys:
                split_metrics[k.lower()] = {
                    "GRU4Rec": float(g_data.get(k, 0.0)),
                    "GRU4RecF": float(f_data.get(k, 0.0)),
                    "GRU4RecF_Rerank": float(r_data.get(k, f_data.get(k, 0.0))),
                }
            combined_splits[(pcount, ds_group)][split_suffix] = split_metrics

    return combined_splits


def generate_modification_plots(pcount: str, ds_group: str, splits_data: dict[str, dict[str, dict]]):
    if not splits_data:
        return

    splits = list(splits_data.keys())
    all_metric_keys = set()
    for s in splits:
        all_metric_keys.update(splits_data[s].keys())

    avg_metrics = {}
    for m in all_metric_keys:
        avg_metrics[m] = {}
        for model in MODIFICATION_MODELS:
            vals = [splits_data[s][m][model] for s in splits if m in splits_data[s] and model in splits_data[s][m]]
            mean_val = float(np.mean(vals)) if vals else 0.0
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            avg_metrics[m][model] = {"mean": mean_val, "std": std_val}

    valid_metric_keys = [
        m for m in all_metric_keys
        if "@" in m and m.split("@")[0].lower() in BASE_ORDER and m.split("@")[1].isdigit() and int(m.split("@")[1]) in ALLOWED_KS
    ]
    sorted_metric_keys = sorted(valid_metric_keys, key=get_metric_sort_key)

    mod_json_aggregated = defaultdict(dict)
    for model in MODIFICATION_MODELS:
        m_label = MODEL_DISPLAY_LABELS.get(model, model)
        for m in sorted_metric_keys:
            if m in avg_metrics and model in avg_metrics[m]:
                mean_val = avg_metrics[m][model]["mean"]
                mod_json_aggregated[m_label][m.lower()] = round(mean_val, 4)

    mod_dir = get_modification_dir(pcount, ds_group)
    mod_bar_dir = get_modification_bar_dir(pcount, ds_group)
    mod_line_dir = get_modification_line_dir(pcount, ds_group)

    mod_dir.mkdir(parents=True, exist_ok=True)
    out_mod_json = mod_dir / "average_results.json"
    try:
        with open(out_mod_json, "w", encoding="utf-8") as f:
            json.dump(dict(mod_json_aggregated), f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    # 1. Bar Plots
    for base_m in METRIC_BASES:
        display_name = METRIC_DISPLAY_NAMES.get(base_m, base_m.upper())
        file_m_base = "hitrate" if base_m == "hit" else base_m
        is_cov = base_m == "itemcoverage"
        is_pop = base_m == "averagepopularity"

        for k in KS:
            key = f"{base_m}@{k}".lower()
            if key not in avg_metrics:
                continue

            vals = []
            stds = []
            display_labels = []

            for model in MODIFICATION_MODELS:
                v = avg_metrics[key][model]["mean"]
                s = avg_metrics[key][model]["std"]
                if is_cov and v <= 1.0:
                    v *= 100
                    s *= 100
                vals.append(v)
                stds.append(s)
                display_labels.append(MODEL_DISPLAY_LABELS.get(model, model))

            red_line_val = vals[MODIFICATION_MODELS.index("GRU4Rec")] if "GRU4Rec" in MODIFICATION_MODELS else None

            unit = " (%)" if is_cov else ""
            y_label = f"{display_name} @ {k}{unit}"
            out_file = mod_bar_dir / f"{file_m_base}@{k}.png"

            render_bar_chart(
                display_labels=display_labels,
                vals=vals,
                stds=stds,
                y_label=y_label,
                output_file=out_file,
                is_coverage=is_cov,
                is_pop=is_pop,
                red_line_value=red_line_val,
            )

    # 2. Line Plots
    for base_m in METRIC_BASES:
        display_name = METRIC_DISPLAY_NAMES.get(base_m, base_m.upper())
        file_m_base = "hitrate" if base_m == "hit" else base_m
        is_cov = base_m == "itemcoverage"

        model_curves = defaultdict(list)
        for model in MODIFICATION_MODELS:
            for k in KS:
                key = f"{base_m}@{k}".lower()
                if key in avg_metrics:
                    v = avg_metrics[key][model]["mean"]
                    if is_cov and v <= 1.0:
                        v *= 100
                    model_curves[model].append(v)

        if not model_curves:
            continue

        unit = " (%)" if is_cov else ""
        y_label = f"{display_name}{unit}"
        out_line_file = mod_line_dir / f"{file_m_base}.png"

        render_line_chart(
            models=MODIFICATION_MODELS,
            model_curves=model_curves,
            ks=KS,
            y_label=y_label,
            output_file=out_line_file,
        )


def format_seconds(seconds: float) -> str:
    secs = int(round(seconds))
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


def parse_model_log_times(model_name: str, ds_group: str, pcount: str = "") -> dict:
    pcount_tag = f"pcount[{pcount}]" if pcount else ""
    candidates = []
    if (LOG_DIR / model_name).is_dir():
        for p in (LOG_DIR / model_name).glob("*.log"):
            if ds_group in p.name and (not pcount_tag or pcount_tag in p.name):
                candidates.append(p)
    for p in LOG_DIR.glob(f"{model_name}-*.log"):
        if ds_group in p.name and (not pcount_tag or pcount_tag in p.name):
            candidates.append(p)
    for p in LOG_DIR.glob(f"{model_name}_*.log"):
        if ds_group in p.name and (not pcount_tag or pcount_tag in p.name):
            candidates.append(p)

    candidates = sorted(list(set(candidates)))

    time_pattern = re.compile(r"epoch\s+\d+\s+training\s+\[time:\s*([\d\.]+)s")
    eval_pattern = re.compile(r"epoch\s+\d+\s+evaluating\s+\[time:\s*([\d\.]+)s")
    timestamp_pattern = re.compile(r"^[A-Z][a-z]{2}\s+\d+\s+[A-Z][a-z]{2}\s+\d{4}\s+(\d{2}:\d{2}:\d{2})")

    runs = []
    for log_path in candidates:
        train_times = []
        eval_times = []
        first_dt = None
        last_dt = None

        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    t_match = time_pattern.search(line)
                    if t_match:
                        train_times.append(float(t_match.group(1)))
                    e_match = eval_pattern.search(line)
                    if e_match:
                        eval_times.append(float(e_match.group(1)))

                    ts_match = timestamp_pattern.match(line)
                    if ts_match:
                        time_str = ts_match.group(1)
                        h, m, s = map(int, time_str.split(":"))
                        total_secs = h * 3600 + m * 60 + s
                        if first_dt is None:
                            first_dt = total_secs
                        last_dt = total_secs
        except Exception:
            continue

        if train_times:
            total_train = sum(train_times)
            num_epochs = len(train_times)
            avg_epoch = total_train / num_epochs
            wall_sec = (last_dt - first_dt) if (first_dt is not None and last_dt is not None) else total_train
            if wall_sec < 0:
                wall_sec += 86400

            runs.append({
                "log_file": log_path.name,
                "train_time_sec": round(total_train, 2),
                "eval_time_sec": round(sum(eval_times), 2),
                "wall_time_sec": round(wall_sec, 2),
                "epochs": num_epochs,
                "avg_epoch_sec": round(avg_epoch, 2),
            })

    if not runs:
        return {}

    train_secs = [r["train_time_sec"] for r in runs]
    wall_secs = [r["wall_time_sec"] for r in runs]
    epoch_secs = [r["avg_epoch_sec"] for r in runs]

    mean_train = float(np.mean(train_secs))
    std_train = float(np.std(train_secs, ddof=1)) if len(train_secs) > 1 else 0.0
    mean_wall = float(np.mean(wall_secs))
    std_wall = float(np.std(wall_secs, ddof=1)) if len(wall_secs) > 1 else 0.0
    mean_epoch = float(np.mean(epoch_secs))

    return {
        "num_runs": len(runs),
        "mean_train_time_sec": round(mean_train, 2),
        "std_train_time_sec": round(std_train, 2),
        "mean_wall_time_sec": round(mean_wall, 2),
        "std_wall_time_sec": round(std_wall, 2),
        "mean_epoch_time_sec": round(mean_epoch, 2),
        "formatted_mean_train_time": format_seconds(mean_train),
        "formatted_total_time": format_seconds(sum(train_secs)),
        "runs": runs,
    }


def generate_training_time_summaries(pcount: str, ds_group: str):
    baseline_dir = get_baseline_dir(pcount, ds_group)
    baseline_bar_dir = get_baseline_bar_dir(pcount, ds_group)
    mod_dir = get_modification_dir(pcount, ds_group)
    mod_bar_dir = get_modification_bar_dir(pcount, ds_group)

    # 1. Baseline Training Times
    baseline_summary = {}
    for model in FIXED_BASELINE_ORDER:
        info = parse_model_log_times(model, ds_group, pcount)
        if info:
            baseline_summary[model] = info

    if baseline_summary:
        baseline_dir.mkdir(parents=True, exist_ok=True)
        with open(baseline_dir / "training_times.json", "w", encoding="utf-8") as f:
            json.dump(baseline_summary, f, indent=2, ensure_ascii=False)

        b_labels = [MODEL_DISPLAY_LABELS.get(m, m) for m in FIXED_BASELINE_ORDER if m in baseline_summary]
        b_vals = [baseline_summary[m]["mean_train_time_sec"] for m in FIXED_BASELINE_ORDER if m in baseline_summary]
        b_stds = [baseline_summary[m]["std_train_time_sec"] for m in FIXED_BASELINE_ORDER if m in baseline_summary]
        if b_vals:
            render_bar_chart(
                display_labels=b_labels,
                vals=b_vals,
                stds=b_stds,
                y_label="Czas uczenia (s)",
                output_file=baseline_bar_dir / "training_time.png",
                is_pop=True,
            )

    # 2. Modification Training Times
    mod_summary = {}
    for model in MODIFICATION_MODELS:
        info = parse_model_log_times(model, ds_group, pcount)
        if info:
            mod_summary[MODEL_DISPLAY_LABELS.get(model, model)] = info

    if mod_summary:
        mod_dir.mkdir(parents=True, exist_ok=True)
        with open(mod_dir / "training_times.json", "w", encoding="utf-8") as f:
            json.dump(mod_summary, f, indent=2, ensure_ascii=False)

        m_labels = [MODEL_DISPLAY_LABELS.get(m, m) for m in MODIFICATION_MODELS if MODEL_DISPLAY_LABELS.get(m, m) in mod_summary]
        m_vals = [mod_summary[lbl]["mean_train_time_sec"] for lbl in m_labels if lbl in mod_summary]
        m_stds = [mod_summary[lbl]["std_train_time_sec"] for lbl in m_labels if lbl in mod_summary]
        if m_vals:
            render_bar_chart(
                display_labels=m_labels,
                vals=m_vals,
                stds=m_stds,
                y_label="Czas uczenia (s)",
                output_file=mod_bar_dir / "training_time.png",
                is_pop=True,
            )

def main():
    raw_results = load_raw_model_results()
    mod_splits = load_modification_data()

    all_keys = sorted(list(set(raw_results.keys()).union(mod_splits.keys())))
    if not all_keys:
        all_keys = [("25", "30music"), ("25", "lastfm1k"), ("5", "30music"), ("5", "lastfm1k")]

    for pcount, ds_group in all_keys:
        print(f"Processing pcount={pcount}, ds_group={ds_group}")
        if (pcount, ds_group) in raw_results:
            generate_baseline_plots(pcount, ds_group, raw_results[(pcount, ds_group)])
        if (pcount, ds_group) in mod_splits:
            generate_modification_plots(pcount, ds_group, mod_splits[(pcount, ds_group)])
        generate_training_time_summaries(pcount, ds_group)


if __name__ == "__main__":
    main()