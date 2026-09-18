import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SAVED_PATH = Path("saved")
RESULTS_PATH = Path("results")

KS = [5, 10, 20]

METRICS = {
    "precision": "Precision", 
    "recall": "Recall", 
    "mrr": "MRR",
    "ndcg": "NDCG", 
    "itemcoverage": "Item Coverage", 
    "averagepopularity": "Average Popularity",
}

BASELINE_MODELS = ["FPMC", "GRU4Rec", "NARM", "STAMP", "SRGNN", "SASRec"]
MODIFICATION_MODELS = ["GRU4Rec", "GRU4RecF", "GRU4RecF+"]

MARKERS = {
    "FPMC": "o", 
    "GRU4Rec": "s", 
    "NARM": "^", 
    "STAMP": "D",
    "SRGNN": "v", 
    "SASRec": "p", 
    "GRU4RecF": "s", 
    "GRU4RecF+": "^",
}

COLORS = ["#356070", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51", "#7209b7", "#4361ee"]

sns.set_palette(COLORS)
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral", "serif"],
    "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

def average_over_splits(splits, models):
    averages = defaultdict(dict)

    for model in models:
        for metric in METRICS:
            for k in KS:
                key = f"{metric}@{k}"

                if isinstance(splits[0][key], dict):
                    values = [
                        split[key][model]
                        for split in splits
                        if key in split and model in split[key]
                    ]
                else:
                    values = [
                        split[key]
                        for split in splits
                        if key in split
                    ]

                if values:
                    averages[model][key] = float(np.mean(values))

    return averages

def save_averages_json(dataset, suffix, models, averages):
    output = {}

    for model in models:
        if model in averages:
            output[model] = {}

            for metric in METRICS:
                for k in KS:
                    key = f"{metric}@{k}"

                    if key in averages[model]:
                        output[model][key] = round(averages[model][key], 4)

    path = RESULTS_PATH / f"{dataset}_{suffix}_average_results.json"

    with open(path, "w", encoding="utf-8") as file_out:
        json.dump(output, file_out, indent=2, ensure_ascii=False)

def plot_grid(models, avg, output_path):
    nrows, ncols = 3, 2

    fig = plt.figure(figsize=(6.0, 8.6), constrained_layout=True)
    gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[0.8] + [2.6] * nrows)

    legend_ax = fig.add_subplot(gs[0, :])
    legend_ax.axis("off")

    axes = [
        fig.add_subplot(gs[row + 1, col])
        for row in range(nrows)
        for col in range(ncols)
    ]

    palette = COLORS[:len(models)]
    handles = {}

    for metric_index, (metric, label) in enumerate(METRICS.items()):
        ax = axes[metric_index]

        for model_index, model in enumerate(models):
            model_avg = avg.get(model, {})
            values = [model_avg.get(f"{metric}@{k}") for k in KS]

            if None in values:
                continue

            if metric == "itemcoverage":
                values = [v * 100 if v <= 1.0 else v for v in values]

            line, = ax.plot(
                KS, values,
                marker=MARKERS.get(model, "o"),
                color=palette[model_index],
                label=model,
                linewidth=1.6,
                markersize=5,
            )

            handles.setdefault(model, line)

        unit = " (%)" if metric == "itemcoverage" else ""
        ax.set_xlabel("K", fontweight="bold", fontsize=10, labelpad=8)
        ax.set_ylabel(f"{label}{unit}", fontweight="bold", fontsize=10, labelpad=8)

        ax.set_xticks(KS)
        ax.set_xticklabels([str(k) for k in KS], fontweight="bold", fontsize=9.5)

        comma_formatter = mpl.ticker.FuncFormatter(
            lambda value, pos: f"{value:g}".replace(".", ",")
        )
        ax.yaxis.set_major_formatter(comma_formatter)
        plt.setp(ax.get_yticklabels(), fontweight="normal", fontsize=9.5)

        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)

    for unused_index in range(len(METRICS), len(axes)):
        axes[unused_index].axis("off")

    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.08, wspace=0.125, hspace=0.125)

    legend_handles = [handles[model] for model in models if model in handles]
    legend_labels = [model for model in models if model in handles]

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=min(len(legend_handles), 4),
            frameon=False,
            fontsize=11,
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

def read_baseline_metrics(path):
    with open(path, "r", encoding="utf-8") as file_in:
        data = json.load(file_in)

    return {
        key.lower(): float(value)
        for key, value in data["test_result"].items()
    }

def load_baseline_results():
    results = defaultdict(lambda: defaultdict(list))

    for folder in sorted(SAVED_PATH.iterdir()):
        if not folder.is_dir():
            continue

        model, dataset = folder.name.split("_", 1)
        dataset = dataset.split("__")[0]
        results_file = folder / "results.json"
        
        if results_file.exists():
            metrics = read_baseline_metrics(results_file)

            if metrics:
                results[dataset][model].append(metrics)
                
    return results

def read_modification_metrics(path):
    with open(path, "r", encoding="utf-8") as file_in:
        data = json.load(file_in)

    baseline = data.get("baseline", {})
    reranking = data.get("reranking", {})

    def to_metrics(data):
        return {
            str(key).lower(): float(value)
            for key, value in data.items()
        }
    
    return to_metrics(baseline), to_metrics(reranking)

def load_modification_results():
    results = defaultdict(list)

    for folder in sorted(SAVED_PATH.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("GRU4Rec_"):
            continue

        suffix = folder.name[len("GRU4Rec_"):]
        dataset = suffix.split("__")[0]
        gru4rec_results = folder / "results.json"

        if not gru4rec_results.exists():
            continue

        gru4rec_metrics = read_baseline_metrics(gru4rec_results)
        gru4recf_results = SAVED_PATH / f"GRU4RecF_{suffix}" / "results_reranking.json"

        if not gru4recf_results.exists():
            continue

        gru4recf_metrics, gru4recf_plus_metrics = read_modification_metrics(gru4recf_results)

        if not gru4recf_metrics:
            continue

        keys = set().union(gru4rec_metrics, gru4recf_metrics, gru4recf_plus_metrics)
        model_metrics = {}
        
        for key in keys:
            model_metrics[key] = {
                "GRU4Rec": gru4rec_metrics.get(key, 0.0),
                "GRU4RecF": gru4recf_metrics.get(key, 0.0),
                "GRU4RecF+": gru4recf_plus_metrics.get(key, gru4recf_metrics.get(key, 0.0)),
            }

        results[dataset].append(model_metrics)

    return results

def generate_baseline(dataset, splits):
    models = [model for model in splits if model != "GRU4RecF"]
    model_order = {name.upper(): i for i, name in enumerate(BASELINE_MODELS)}
    models.sort(key=lambda model: model_order[model.upper()])

    averages = defaultdict(dict)
    for model in models:
        for metric in METRICS:
            for k in KS:
                key = f"{metric}@{k}"
                values = [split[key] for split in splits[model] if key in split]

                if values:
                    averages[model][key] = float(np.mean(values))

    save_averages_json(dataset, "baseline", models, averages)
    plot_grid(models, averages, RESULTS_PATH / f"{dataset}_baseline_grid.png")

def generate_modification(dataset, splits):
    averages = defaultdict(dict)

    for model in MODIFICATION_MODELS:
        for metric in METRICS:
            for k in KS:
                key = f"{metric}@{k}"
                values = [split[key][model] for split in splits if key in split and model in split[key]]

                if values:
                    averages[model][key] = float(np.mean(values))

    save_averages_json(dataset, "modification", MODIFICATION_MODELS, averages)
    plot_grid(MODIFICATION_MODELS, averages, RESULTS_PATH / f"{dataset}_modification_grid.png")

if __name__ == "__main__":
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    baseline = load_baseline_results()
    modification = load_modification_results()

    for dataset in sorted(set(baseline) | set(modification)):
        if dataset in baseline:
            generate_baseline(dataset, baseline[dataset])
        if dataset in modification:
            generate_modification(dataset, modification[dataset])