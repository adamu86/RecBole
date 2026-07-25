import json
import os
from pathlib import Path
from collections import defaultdict

SAVED_DIR = Path("saved")
RESULTS_FILES = ["results_1.json", "results_N.json"]


def load_results(folder: Path) -> dict:
    results = {}
    for filename in RESULTS_FILES:
        filepath = folder / filename
        if filepath.exists():
            with open(filepath, "r") as f:
                data = json.load(f)
            prefix = filename.replace(".json", "")
            test_result = data.get("test_result", {})
            for metric, value in test_result.items():
                results[f"{prefix}/{metric}"] = value
    return results


def generate_comparison_plots(averaged: dict, output_dir: Path):
    """Generate publication-quality comparison plots for Master's thesis."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)

    # Style configuration for thesis
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "STIXGeneral", "Times New Roman"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    models = list(averaged.keys())
    if not models:
        return

    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
    model_colors = {m: colors[i % len(colors)] for i, m in enumerate(models)}

    # Helper function to get metric value
    def get_val(model_name, metric_key):
        metrics = averaged[model_name]["averaged_metrics"]
        return metrics.get(f"results_1/{metric_key}", 0.0)

    # =========================================================================
    # Wykres 1: Dokładność (Recall & Hit na Top-5, 10, 15, 20)
    # =========================================================================
    ks = [5, 10, 15, 20]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    x = np.arange(len(ks))
    width = 0.35 if len(models) == 2 else 0.8 / len(models)

    for i, model in enumerate(models):
        recalls = [get_val(model, f"recall@{k}") for k in ks]
        offset = (i - (len(models) - 1) / 2) * width
        rects = ax.bar(x + offset, recalls, width, label=model, color=model_colors[model], alpha=0.85, edgecolor="black", linewidth=0.5)
        
        # Annotate bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_title("Porównanie Trafności (Recall@K) po Splitach Czasowych", fontweight="bold", pad=12)
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Recall@K")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{k}" for k in ks])
    ax.set_ylim(0, max([get_val(m, "recall@20") for m in models] or [0.4]) * 1.15)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(output_dir / f"1_recall_comparison.{fmt}")
    plt.close(fig)

    # =========================================================================
    # Wykres 2: Różnorodność i Popularity Bias (AvgPopularity, Coverage, Gini)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.2))

    # Panel A: Average Popularity @ 10 (Im niżej, tym mniejszy Popularity Bias)
    pops = [get_val(m, "averagepopularity@10") for m in models]
    x_m = np.arange(len(models))
    bars1 = ax1.bar(x_m, pops, width=0.5, color=[model_colors[m] for m in models], alpha=0.85, edgecolor="black", linewidth=0.5)
    ax1.set_title("Average Popularity@10\n(Niżej = mniejszy Popularity Bias)", fontweight="bold", fontsize=10)
    ax1.set_xticks(x_m)
    ax1.set_xticklabels(models)
    ax1.set_ylabel("Średnia Popularność")
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    # Panel B: Item Coverage @ 10 (Wyżej = lepsze pokrycie katalogu)
    covs = [get_val(m, "itemcoverage@10") * 100 for m in models]
    bars2 = ax2.bar(x_m, covs, width=0.5, color=[model_colors[m] for m in models], alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.set_title("Item Coverage@10\n(Wyżej = większe pokrycie katalogu)", fontweight="bold", fontsize=10)
    ax2.set_xticks(x_m)
    ax2.set_xticklabels(models)
    ax2.set_ylabel("Pokrycie Katalogu (%)")
    ax2.set_ylim(0, 105)
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(output_dir / f"2_diversity_and_coverage.{fmt}")
    plt.close(fig)

    # =========================================================================
    # Wykres 3: 4-Panelowy Raport Zbiorczy do Pracy Magisterskiej
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0))
    ((ax_rec, ax_rank), (ax_pop, ax_cov)) = axes

    # A: Recall@K
    for m in models:
        vals = [get_val(m, f"recall@{k}") for k in ks]
        ax_rec.plot(ks, vals, marker="o", linewidth=1.8, label=m, color=model_colors[m])
    ax_rec.set_title("(A) Recall@K (Trafność)", fontweight="bold", fontsize=10)
    ax_rec.set_xlabel("K")
    ax_rec.set_ylabel("Recall")
    ax_rec.set_xticks(ks)
    ax_rec.legend(fontsize=8)

    # B: Ranking (MRR & NDCG)
    metrics_rank = ["mrr@10", "ndcg@10", "map@10"]
    x_r = np.arange(len(metrics_rank))
    for i, m in enumerate(models):
        r_vals = [get_val(m, r) for r in metrics_rank]
        offset = (i - (len(models) - 1) / 2) * width
        ax_rank.bar(x_r + offset, r_vals, width, label=m, color=model_colors[m], alpha=0.85)
    ax_rank.set_title("(B) Metryki Rangi @ 10", fontweight="bold", fontsize=10)
    ax_rank.set_xticks(x_r)
    ax_rank.set_xticklabels(["MRR@10", "NDCG@10", "MAP@10"])
    ax_rank.set_ylabel("Wynik")
    ax_rank.legend(fontsize=8)

    # C: Popularity Bias
    for m in models:
        p_vals = [get_val(m, f"averagepopularity@{k}") for k in ks]
        ax_pop.plot(ks, p_vals, marker="s", linestyle="--", linewidth=1.8, label=m, color=model_colors[m])
    ax_pop.set_title("(C) Popularity Bias@K (Niżej = lepiej)", fontweight="bold", fontsize=10)
    ax_pop.set_xlabel("K")
    ax_pop.set_ylabel("Średnia Popularność")
    ax_pop.set_xticks(ks)
    ax_pop.legend(fontsize=8)

    # D: Pokrycie katalogu
    for m in models:
        c_vals = [get_val(m, f"itemcoverage@{k}") * 100 for k in ks]
        ax_cov.plot(ks, c_vals, marker="^", linewidth=1.8, label=m, color=model_colors[m])
    ax_cov.set_title("(D) Pokrycie Katalogu@K (%)", fontweight="bold", fontsize=10)
    ax_cov.set_xlabel("K")
    ax_cov.set_ylabel("Item Coverage (%)")
    ax_cov.set_xticks(ks)
    ax_cov.legend(fontsize=8)

    plt.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(output_dir / f"3_master_thesis_overview.{fmt}")
    plt.close(fig)

    print(f"📊 Wykresy porównawcze zapisano w katalogu: {output_dir}")


def main():
    # Model -> Dataset Name -> Results Dict
    raw_model_results: dict[str, dict[str, dict]] = defaultdict(dict)

    for folder in sorted(SAVED_DIR.iterdir()):
        if not folder.is_dir():
            continue
        parts = folder.name.split("_", 1)
        model_name = parts[0]
        dataset_name = parts[1] if len(parts) > 1 else folder.name

        results = load_results(folder)
        if results:
            raw_model_results[model_name][dataset_name] = results

    if not raw_model_results:
        print("Brak wyników w katalogu saved/.")
        return

    # Znajdź wspólne zbiory (iloczyn zbiorów danych dostępnych dla wszystkich modeli)
    dataset_sets = [set(datasets.keys()) for datasets in raw_model_results.values()]
    common_datasets = sorted(list(set.intersection(*dataset_sets)))

    print("=" * 100)
    print("ANALIZA WSPÓLNYCH ZBIORÓW DANYCH (COMMON DATASET SPLITS)")
    print("=" * 100)
    print(f"Wszystkie modele: {list(raw_model_results.keys())}")
    print(f"Liczba wspólnych zbiorów: {len(common_datasets)}")
    print("Wspólne zbiory:")
    for ds in common_datasets:
        print(f"  • {ds}")

    # Filtrujemy wyniki tylko do wspólnych zbiorów
    model_results: dict[str, list[dict]] = defaultdict(list)
    for model_name, datasets in raw_model_results.items():
        for ds_name in common_datasets:
            if ds_name in datasets:
                model_results[model_name].append({
                    "split": f"{model_name}_{ds_name}",
                    "metrics": datasets[ds_name]
                })

    averaged = {}
    for model_name, splits in sorted(model_results.items()):
        all_metrics = set()
        for split in splits:
            all_metrics.update(split["metrics"].keys())

        avg_metrics = {}
        for metric in sorted(all_metrics):
            values = [s["metrics"][metric] for s in splits if metric in s["metrics"]]
            if values:
                avg_metrics[metric] = round(sum(values) / len(values), 4)

        averaged[model_name] = avg_metrics

    output_path = SAVED_DIR / "average_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(averaged, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("UŚREDNIONE WYNIKI MODELI (po splitach czasowych)")
    print("=" * 100)

    for model_name, data in sorted(averaged.items()):
        print(f"\n{'─' * 80}")
        print(f"  Model: {model_name}  |  Liczba splitów: {data['num_splits']}")
        print(f"{'─' * 80}")

        # Podziel metryki na results_1 i results_N
        for prefix in ["results_1", "results_N"]:
            prefix_metrics = {
                k.split("/", 1)[1]: v
                for k, v in data["averaged_metrics"].items()
                if k.startswith(prefix)
            }
            if not prefix_metrics:
                continue

            label = "Next-Item (results_1)" if prefix == "results_1" else "Next-N-Items (results_N)"
            print(f"\n  {label}:")
            print(f"  {'Metryka':<30} {'Średnia':>10}")
            print(f"  {'─' * 42}")
            for metric, value in sorted(prefix_metrics.items()):
                print(f"  {metric:<30} {value:>10.4f}")

    print(f"\n{'=' * 100}")
    print(f"Wyniki zapisane do: {output_path}")
    print(f"{'=' * 100}")

    # Generowanie wykresów porównawczych
    plots_dir = SAVED_DIR / "comparison_plots"
    generate_comparison_plots(averaged, plots_dir)


if __name__ == "__main__":
    main()
