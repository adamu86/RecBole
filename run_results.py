import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SAVED_DIR = Path("saved")
RESULTS_DIR = Path("results")
KS = [5, 10, 20]
METRICS = {
    "precision": "Precision", "recall": "Recall", "mrr": "MRR",
    "ndcg": "NDCG", "itemcoverage": "Item Coverage", "averagepopularity": "Average Popularity",
}
BASELINE_MODELS = ["FPMC", "GRU4Rec", "NARM", "STAMP", "SRGNN", "SASRec"]
MODIFICATION_MODELS = ["GRU4Rec", "GRU4RecF", "GRU4RecF+"]

MARKERS = {
    "FPMC": "o", "GRU4Rec": "s", "NARM": "^", "STAMP": "D",
    "SRGNN": "v", "SASRec": "p", "GRU4RecF": "s", "GRU4RecF+": "^",
}

COLORS = ["#356070", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51", "#7209b7", "#4361ee"]
sns.set_palette(COLORS)
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral", "serif"],
    "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

def read_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    res = data.get("test_result", data)
    if isinstance(res, dict):
        return {str(k).lower(): float(v) for k, v in res.items() if isinstance(v, (int, float))}
    return {}

def read_comparison(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    baseline = data.get("baseline", {})
    reranking = data.get("reranking", {})
    to_dict = lambda d: {str(k).lower(): float(v) for k, v in d.items() if isinstance(v, (int, float))}
    return to_dict(baseline), to_dict(reranking)

def average_over_splits(splits, models):
    avg = defaultdict(dict)
    for model in models:
        for m in METRICS:
            for k in KS:
                key = f"{m}@{k}"
                vals = [s[key] for s in splits if key in s] if isinstance(splits[0], dict) and key in splits[0] and not isinstance(splits[0][key], dict) else \
                       [s[key][model] for s in splits if key in s and model in s[key]]
                if vals:
                    avg[model][key] = float(np.mean(vals))
    return avg

def save_averages_json(ds_group, tag, models, avg):
    out = {}
    for model in models:
        if model not in avg:
            continue
        out[model] = {f"{m}@{k}": round(avg[model][f"{m}@{k}"], 4)
                      for m in METRICS for k in KS if f"{m}@{k}" in avg[model]}
    path = RESULTS_DIR / f"{ds_group}_{tag}_average_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")


def plot_grid(models, avg, output_path):
    nrows, ncols = 3, 2
    fig = plt.figure(figsize=(6.0, 8.6), constrained_layout=True)
    gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[0.8] + [2.6] * nrows)

    legend_ax = fig.add_subplot(gs[0, :])
    legend_ax.axis("off")
    axes = [fig.add_subplot(gs[r + 1, c]) for r in range(nrows) for c in range(ncols)]

    palette = COLORS[:len(models)]
    handles = {}

    for i, (metric, label) in enumerate(METRICS.items()):
        ax = axes[i]
        for j, model in enumerate(models):
            vals = []
            for k in KS:
                key = f"{metric}@{k}"
                if model in avg and key in avg[model]:
                    v = avg[model][key]
                    if metric == "itemcoverage" and v <= 1.0:
                        v *= 100
                    vals.append(v)

            if len(vals) == len(KS):
                line, = ax.plot(KS, vals, marker=MARKERS.get(model, "o"),
                                color=palette[j], label=model,
                                linewidth=1.6, markersize=5)
                if model not in handles:
                    handles[model] = line

        unit = " (%)" if metric == "itemcoverage" else ""
        ax.set_xlabel("K", fontweight="bold", fontsize=10, labelpad=8)
        ax.set_ylabel(f"{label}{unit}", fontweight="bold", fontsize=10, labelpad=8)
        ax.set_xticks(KS)
        ax.set_xticklabels([str(k) for k in KS], fontweight="bold", fontsize=9.5)
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, p: f"{v:g}".replace(".", ",")))
        plt.setp(ax.get_yticklabels(), fontweight="normal", fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)

    for i in range(len(METRICS), len(axes)):
        axes[i].axis("off")

    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.08, wspace=0.125, hspace=0.125)

    h = [handles[m] for m in models if m in handles]
    l = [m for m in models if m in handles]
    if h:
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                   ncol=min(len(h), 4), frameon=False, fontsize=11)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# --- Ładowanie wyników ---

def load_baseline_results():
    results = defaultdict(lambda: defaultdict(list))
    if not SAVED_DIR.exists():
        return results

    for folder in sorted(SAVED_DIR.iterdir()):
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        parts = folder.name.split("_", 1)
        if len(parts) < 2:
            continue

        model, dataset = parts
        ds_group = dataset.split("__")[0]

        res_file = folder / "results.json"
        if res_file.exists():
            try:
                metrics = read_metrics(res_file)
                if metrics:
                    results[ds_group][model].append(metrics)
            except Exception:
                pass
    return results


def load_modification_results():
    results = defaultdict(list)
    if not SAVED_DIR.exists():
        return results

    for folder in sorted(SAVED_DIR.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("GRU4Rec_"):
            continue

        suffix = folder.name[len("GRU4Rec_"):]
        ds_group = suffix.split("__")[0]

        g_file = folder / "results.json"
        if not g_file.exists():
            continue
        try:
            g_data = read_metrics(g_file)
        except Exception:
            continue
        if not g_data:
            continue

        # GRU4RecF + reranking
        f_dir = SAVED_DIR / f"GRU4RecF_{suffix}"
        rerank_file = f_dir / "results_reranking.json"

        if not rerank_file.exists():
            continue

        try:
            f_data, r_data = read_comparison(rerank_file)
        except Exception:
            continue

        if not f_data:
            continue

        all_keys = set(g_data) | set(f_data) | set(r_data)
        split = {}
        for key in all_keys:
            split[key] = {
                "GRU4Rec": g_data.get(key, 0.0),
                "GRU4RecF": f_data.get(key, 0.0),
                "GRU4RecF+": r_data.get(key, f_data.get(key, 0.0)),
            }
        results[ds_group].append(split)

    return results


# --- Generowanie wykresów ---

def generate_baseline(ds_group, model_splits):
    models = [m for m in model_splits if m.upper() != "GRU4RECF"]
    order = {name.upper(): i for i, name in enumerate(BASELINE_MODELS)}
    models.sort(key=lambda m: order.get(m.upper(), 999))
    if not models:
        return

    avg = defaultdict(dict)
    for model in models:
        for m in METRICS:
            for k in KS:
                key = f"{m}@{k}"
                vals = [s[key] for s in model_splits[model] if key in s]
                if vals:
                    avg[model][key] = float(np.mean(vals))

    ordered = sorted(avg.keys(), key=lambda m: order.get(m.upper(), 999))
    save_averages_json(ds_group, "baseline", ordered, avg)
    plot_grid(models, avg, RESULTS_DIR / f"{ds_group}_baseline_grid.png")

def generate_modification(ds_group, splits):
    if not splits:
        return

    avg = defaultdict(dict)
    for model in MODIFICATION_MODELS:
        for m in METRICS:
            for k in KS:
                key = f"{m}@{k}"
                vals = [s[key][model] for s in splits if key in s and model in s[key]]
                if vals:
                    avg[model][key] = float(np.mean(vals))

    save_averages_json(ds_group, "modification", MODIFICATION_MODELS, avg)
    plot_grid(MODIFICATION_MODELS, avg, RESULTS_DIR / f"{ds_group}_modification_grid.png")

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline = load_baseline_results()
    modification = load_modification_results()

    for ds_group in sorted(set(baseline) | set(modification)):
        print(f"Processing: {ds_group}")
        if ds_group in baseline:
            generate_baseline(ds_group, baseline[ds_group])
        if ds_group in modification:
            generate_modification(ds_group, modification[ds_group])