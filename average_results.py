import argparse
import json
import os
import sys
import statistics
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
                if prefix == "results_1":
                    results[metric] = value
                else:
                    results[f"{prefix}/{metric}"] = value
    return results


def select_models(available_models: list[str], models_arg: list[str] | None = None, force_interactive: bool = False) -> list[str]:
    if not available_models:
        return []

    # 1. Jeżeli podano modele jako argumenty CLI
    if models_arg:
        requested = []
        for item in models_arg:
            for name in item.replace(",", " ").split():
                if name:
                    requested.append(name)
        
        matched = []
        for r in requested:
            for avail in available_models:
                if r.lower() == avail.lower() and avail not in matched:
                    matched.append(avail)

        if matched:
            return matched
        else:
            print(f"⚠️ Nie znaleziono podanych modeli: {models_arg}. Dostępne modele w saved/: {available_models}")

    # 2. Jeśli brak TTY (środowisko nieinteraktywne) i nie wymuszono interakcji, użyj wszystkich
    if not force_interactive and not sys.stdin.isatty():
        return available_models

    # 3. Interaktywne menu wyboru
    print("\n" + "=" * 80)
    print("WYBÓR MODELI DO UŚREDNIENIA WYNIKÓW")
    print("=" * 80)
    for idx, model in enumerate(available_models, 1):
        print(f"  [{idx}] {model}")
    print(f"  [0] Wszystkie modele ({', '.join(available_models)})")
    print("-" * 80)

    try:
        user_input = input("Wybierz modele (np. '1, 3' lub 'GRU4Rec SASRec', Enter = wszystkie): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nUżywanie wszystkich dostępnych modeli.")
        return available_models

    if not user_input or user_input == "0" or user_input.lower() in ["all", "wszystkie", "*"]:
        return available_models

    selected = []
    tokens = user_input.replace(",", " ").split()
    for token in tokens:
        if token.isdigit():
            idx = int(token)
            if 1 <= idx <= len(available_models):
                model_name = available_models[idx - 1]
                if model_name not in selected:
                    selected.append(model_name)
            else:
                print(f"⚠️ Numer {idx} jest poza zakresem (1-{len(available_models)})")
        else:
            matched = False
            for avail in available_models:
                if token.lower() == avail.lower():
                    if avail not in selected:
                        selected.append(avail)
                    matched = True
                    break
            if not matched:
                print(f"⚠️ Nie rozpoznano nazwy modelu: '{token}'")

    if not selected:
        print("Nie wybrano prawidłowych modeli. Używanie wszystkich dostępnych modeli.")
        return available_models

    return selected


def select_metrics(available_metrics: list[str], metrics_arg: list[str] | None = None, force_interactive: bool = False) -> list[str]:
    if not available_metrics:
        return []

    # 1. CLI argument check
    if metrics_arg:
        requested = []
        for item in metrics_arg:
            for name in item.replace(",", " ").split():
                if name:
                    requested.append(name.lower())
        
        group_map = {
            "a": ["recall", "precision", "hit"],
            "b": ["mrr", "ndcg", "map"],
            "c": ["itemcoverage", "giniindex", "averagepopularity"],
            "accuracy": ["recall", "precision", "hit"],
            "ranking": ["mrr", "ndcg", "map"],
            "diversity": ["itemcoverage", "giniindex", "averagepopularity"],
        }

        matched = []
        for r in requested:
            if r in group_map:
                for base_m in group_map[r]:
                    for avail in available_metrics:
                        if (avail.lower().startswith(f"{base_m}@") or avail.lower() == base_m) and avail not in matched:
                            matched.append(avail)
            else:
                for avail in available_metrics:
                    avail_low = avail.lower()
                    if r == avail_low or avail_low.startswith(f"{r}@") or r in avail_low:
                        if avail not in matched:
                            matched.append(avail)

        if matched:
            return sorted(matched)
        else:
            print(f"⚠️ Nie znaleziono podanych metryk: {metrics_arg}. Dostępne metryki: {available_metrics}")

    # 2. Non-interactive check (brak TTY i nie wymuszono interakcji)
    if not force_interactive and not sys.stdin.isatty():
        return available_metrics

    # Podstawowe typy metryk
    base_metrics = sorted(list(set(m.split("@")[0] for m in available_metrics if "@" in m)))

    print("\n" + "=" * 80)
    print("WYBÓR METRYK DO EWALUACJI I RAPORTU")
    print("=" * 80)
    
    print("  Główne grupy metryk:")
    print("   [A] Metryki Trafności (recall, precision, hit)")
    print("   [B] Metryki Rangi (mrr, ndcg, map)")
    print("   [C] Metryki Różnorodności i Popularności (itemcoverage, giniindex, averagepopularity)")
    print("-" * 80)

    print("  Dostępne typy metryk:")
    for idx, b_metric in enumerate(base_metrics, 1):
        matching_count = sum(1 for m in available_metrics if m.startswith(f"{b_metric}@") or m == b_metric)
        print(f"   [{idx}] {b_metric:<20} (wszystkie @K, razem: {matching_count})")

    print("   [0] Wszystkie metryki")
    print("-" * 80)

    try:
        user_input = input("Wybierz metryki (np. '1, 4', 'recall ndcg', 'A', Enter = wszystkie): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nUżywanie wszystkich dostępnych metryk.")
        return available_metrics

    if not user_input or user_input == "0" or user_input.lower() in ["all", "wszystkie", "*"]:
        return available_metrics

    selected = []
    tokens = user_input.replace(",", " ").split()

    group_map = {
        "a": ["recall", "precision", "hit"],
        "b": ["mrr", "ndcg", "map"],
        "c": ["itemcoverage", "giniindex", "averagepopularity"]
    }

    for token in tokens:
        token_low = token.lower()
        if token_low in group_map:
            for base_m in group_map[token_low]:
                for avail in available_metrics:
                    if (avail.lower().startswith(f"{base_m}@") or avail.lower() == base_m) and avail not in selected:
                        selected.append(avail)
        elif token.isdigit():
            idx = int(token)
            if 1 <= idx <= len(base_metrics):
                target_base = base_metrics[idx - 1]
                for avail in available_metrics:
                    if (avail.lower().startswith(f"{target_base}@") or avail.lower() == target_base) and avail not in selected:
                        selected.append(avail)
            else:
                print(f"⚠️ Numer {idx} jest poza zakresem (1-{len(base_metrics)})")
        else:
            matched = False
            for avail in available_metrics:
                avail_low = avail.lower()
                if token_low == avail_low or avail_low.startswith(f"{token_low}@") or token_low in avail_low:
                    if avail not in selected:
                        selected.append(avail)
                    matched = True
            if not matched:
                print(f"⚠️ Nie rozpoznano metryki ani grupy: '{token}'")

    if not selected:
        print("Nie wybrano prawidłowych metryk. Używanie wszystkich dostępnych metryk.")
        return available_metrics

    return sorted(selected)


import math

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

    custom_colors = {
        "GRU4Rec": "#1f4e79",   # Ciemnoniebieski (Granatowy)
        "GRU4RecF": "#5b9bd5",  # Jasnoniebieski / Błękitny (jaśniejszy od GRU4Rec)
        "SASRec": "#d95f02",    # Pomarańczowy
        "SASRecF": "#e7298a",   # Różowy
        "SRGNN": "#2ca02c",     # Zielony
        "STAMP": "#7570b3",     # Fioletowy
        "NARM": "#8c564b",      # Brązowy
    }
    fallback_colors = ["#1f4e79", "#5b9bd5", "#d95f02", "#7570b3", "#e7298a", "#2ca02c", "#8c564b", "#bcbd22", "#17becf"]
    model_colors = {
        m: custom_colors.get(m, fallback_colors[i % len(fallback_colors)])
        for i, m in enumerate(models)
    }

    # Helper functions to get metric value and std (bez prefiksu results_1)
    def get_val(model_name, metric_key):
        metrics = averaged[model_name]
        val = metrics.get(metric_key, metrics.get(f"results_1/{metric_key}", metrics.get(f"results_N/{metric_key}", 0.0)))
        if isinstance(val, dict):
            return val.get("mean", 0.0)
        return float(val)

    def get_std(model_name, metric_key):
        metrics = averaged[model_name]
        val = metrics.get(metric_key, metrics.get(f"results_1/{metric_key}", metrics.get(f"results_N/{metric_key}", 0.0)))
        if isinstance(val, dict):
            return val.get("std", 0.0)
        return 0.0

    ks = [5, 10, 15, 20]

    all_metrics_keys = set()
    for m in models:
        all_metrics_keys.update(averaged[m].keys())

    base_metric_names = sorted(list(set(k.split("@")[0] for k in all_metrics_keys if "@" in k)))

    metric_display_names = {
        "recall": "Recall",
        "ndcg": "NDCG",
        "mrr": "MRR",
        "map": "MAP",
        "precision": "Precision",
        "hit": "Hit Ratio",
        "averagepopularity": "Average Popularity",
        "itemcoverage": "Item Coverage (%)",
        "giniindex": "Gini Index",
    }

    def safe_savefig(fig_obj, filepath: Path):
        targets = [
            filepath,
            filepath.parent / f"{filepath.stem}_new{filepath.suffix}",
            filepath.parent.parent / "plots" / filepath.name,
            Path("plots") / filepath.name,
        ]

        for target in targets:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    try:
                        import os
                        os.chmod(target, 0o777)
                    except Exception:
                        pass
                    try:
                        target.unlink()
                    except Exception:
                        pass
                fig_obj.savefig(target)
                if target != filepath:
                    print(f"⚠️ Zapisano wykres do ścieżki alternatywnej: '{target}' (oryginalny katalog zablokowany).")
                return
            except PermissionError:
                continue
            except Exception as e:
                print(f"⚠️ Błąd podczas zapisywania '{target}': {e}")
                return

        print(f"⚠️ Pominięto zapis wykresu '{filepath.name}' z powodu braku uprawnień zapisu w systemie plików.")

    # =========================================================================
    # 1. Wykresy zbiorcze dla każdej bazowej metryki (@5, @10, @15, @20)
    # =========================================================================
    for base_m in base_metric_names:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        x = np.arange(len(ks))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            vals = [get_val(model, f"{base_m}@{k}") for k in ks]
            stds = [get_std(model, f"{base_m}@{k}") for k in ks]
            if base_m == "itemcoverage":
                vals = [v * 100 for v in vals]
                stds = [s * 100 for s in stds]

            offset = (i - (len(models) - 1) / 2) * width
            label_text = f"{model}"
            rects = ax.bar(x + offset, vals, width, yerr=stds, capsize=3, label=label_text, color=model_colors[model], alpha=0.85, edgecolor="black", linewidth=0.5)

            for k_idx, (rect, val, std) in enumerate(zip(rects, vals, stds)):
                height = rect.get_height()
                fmt_val = f"{height:.1f}" if base_m == "averagepopularity" else (f"{height:.1f}%" if base_m == "itemcoverage" else f"{height:.3f}")
                ax.annotate(fmt_val,
                            xy=(rect.get_x() + rect.get_width() / 2, height + std),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        title_name = metric_display_names.get(base_m, base_m.upper())
        ax.set_title(title_name, fontweight="bold", fontsize=14, pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"@{k}" for k in ks], fontweight="bold", fontsize=11)
        
        all_v = []
        for m in models:
            for k in ks:
                v = get_val(m, f"{base_m}@{k}")
                if base_m == "itemcoverage":
                    v *= 100
                all_v.append(v + get_std(m, f"{base_m}@{k}") * (100 if base_m == "itemcoverage" else 1))
        
        max_y = max(all_v) if all_v else 1.0
        ax.set_ylim(0, max_y * 1.30)
        ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="lightgray", fontsize=9.5)
        plt.tight_layout()

        safe_savefig(fig, output_dir / f"metric_{base_m}_comparison.png")
        plt.close(fig)

    # =========================================================================
    # 2. Indywidualne wykresy dla KAŻDEGO K (Posortowane od najniższego do najwyższego)
    # =========================================================================
    for base_m in base_metric_names:
        display_name = metric_display_names.get(base_m, base_m.upper())
        for k in ks:
            metric_key = f"{base_m}@{k}"
            
            model_data = []
            for model in models:
                v = get_val(model, metric_key)
                s = get_std(model, metric_key)
                if base_m == "itemcoverage":
                    v *= 100
                    s *= 100
                model_data.append((model, v, s))

            # Sortowanie od najniższej do najwyższej wartości
            sorted_data = sorted(model_data, key=lambda x: x[1])

            sorted_models = [x[0] for x in sorted_data]
            sorted_vals = [x[1] for x in sorted_data]
            sorted_stds = [x[2] for x in sorted_data]

            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            x_pos = np.arange(len(sorted_models))
            bar_colors = [model_colors[m] for m in sorted_models]

            rects = ax.bar(x_pos, sorted_vals, yerr=sorted_stds, capsize=4, color=bar_colors, alpha=0.88, edgecolor="black", linewidth=0.6, width=0.55)

            for rect, val, std in zip(rects, sorted_vals, sorted_stds):
                height = rect.get_height()
                if base_m == "averagepopularity":
                    fmt_val = f"{height:.1f}"
                elif base_m == "itemcoverage":
                    fmt_val = f"{height:.1f}%"
                else:
                    fmt_val = f"{height:.4f}"

                ax.annotate(fmt_val,
                            xy=(rect.get_x() + rect.get_width() / 2, height + std),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=10, fontweight="bold")

            ax.set_title(f"{display_name} @ {k}", fontweight="bold", fontsize=14, pad=12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(sorted_models, fontweight="bold", fontsize=10.5)

            max_y = max([v + s for v, s in zip(sorted_vals, sorted_stds)]) if sorted_vals else 1.0
            ax.set_ylim(0, max_y * 1.25)
            plt.tight_layout()

            safe_savefig(fig, output_dir / f"metric_{base_m}@{k}_comparison.png")
            plt.close(fig)

    print(f"📊 Wykresy porównawcze (indywidualne dla każdego K oraz zbiorcze) zapisano w formacie PNG w katalogu: {output_dir}")





def main():
    parser = argparse.ArgumentParser(description="Uśrednianie wyników ewaluacji wybranych modeli wraz z odchyleniem standardowym.")
    parser.add_argument("-m", "--models", nargs="+", help="Lista modeli do uśrednienia (np. -m GRU4Rec SASRec lub -m GRU4Rec,SASRec)")
    parser.add_argument("-k", "--metrics", nargs="+", help="Lista metryk do uwzględnienia (np. -k recall ndcg lub -k recall@10 averagepopularity@10 lub -k A B)")
    parser.add_argument("-i", "--interactive", action="store_true", help="Uruchom w trybie interaktywnego wyboru modeli i metryk")
    parser.add_argument("-o", "--output", type=Path, default=SAVED_DIR / "average_results.json", help="Ścieżka pliku wynikowego JSON")
    parser.add_argument("--plots-dir", type=Path, default=SAVED_DIR / "comparison_plots", help="Katalog dla wykresów porównawczych")
    args = parser.parse_args()

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

    available_models = sorted(list(raw_model_results.keys()))

    # Wybór modeli
    selected_models = select_models(available_models, args.models, args.interactive)
    print(f"\nWybrane modele do uśrednienia: {', '.join(selected_models)}")

    # Znajdź wspólne zbiory danych dla wybranych modeli
    dataset_sets = [set(raw_model_results[m].keys()) for m in selected_models if m in raw_model_results]
    if dataset_sets:
        common_datasets = sorted(list(set.intersection(*dataset_sets)))
    else:
        common_datasets = []

    print("=" * 100)
    print("ANALIZA WSPÓLNYCH ZBIORÓW DANYCH (COMMON DATASET SPLITS)")
    print("=" * 100)
    print(f"Liczba wspólnych zbiorów dla wybranych modeli: {len(common_datasets)}")
    if common_datasets:
        print("Wspólne zbiory:")
        for ds in common_datasets:
            print(f"  • {ds}")
    else:
        print("⚠️ Uwaga: Wybrane modele nie posiadają nakładających się zbiorów danych. Wyniki zostaną uśrednione po wszystkich dostępnych splitach dla każdego z wybranego modelu.")

    # Filtrujemy wyniki
    model_results: dict[str, list[dict]] = defaultdict(list)
    all_available_metrics = set()
    for model_name in selected_models:
        if model_name not in raw_model_results:
            continue
        datasets = raw_model_results[model_name]
        target_datasets = common_datasets if common_datasets else sorted(list(datasets.keys()))
        for ds_name in target_datasets:
            if ds_name in datasets:
                model_results[model_name].append({
                    "split": f"{model_name}_{ds_name}",
                    "metrics": datasets[ds_name]
                })
                all_available_metrics.update(datasets[ds_name].keys())

    sorted_available_metrics = sorted(list(all_available_metrics))

    # Wybór metryk
    selected_metrics = select_metrics(sorted_available_metrics, args.metrics, args.interactive)
    print(f"\nWybrane metryki do raportowania: {', '.join(selected_metrics)}")

    averaged = {}
    for model_name, splits in sorted(model_results.items()):
        if not splits:
            continue
        all_metrics = set()
        for split in splits:
            all_metrics.update(split["metrics"].keys())

        avg_metrics = {}
        for metric in sorted(all_metrics):
            if selected_metrics and metric not in selected_metrics:
                continue
            values = [s["metrics"][metric] for s in splits if metric in s["metrics"]]
            if values:
                mean_val = sum(values) / len(values)
                if len(values) > 1:
                    variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)

                    std_val = math.sqrt(variance)
                else:
                    std_val = 0.0
                
                avg_metrics[metric] = {
                    "mean": round(mean_val, 4),
                    "std": round(std_val, 4),
                    "display": f"{mean_val:.4f} ± {std_val:.4f}"
                }

        averaged[model_name] = avg_metrics

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(averaged, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("UŚREDNIONE WYNIKI MODELI (ŚREDNIA ± ODCHYLENIE STANDARDOWE)")
    print("=" * 100)

    for model_name, data in sorted(averaged.items()):
        num_splits = len(model_results[model_name])
        print(f"\n{'─' * 85}")
        print(f"  Model: {model_name}  |  Liczba splitów: {num_splits}")
        print(f"{'─' * 85}")

        has_prefixes = any("/" in k for k in data.keys())
        if has_prefixes:
            prefixes = sorted(list(set(k.split("/", 1)[0] for k in data.keys() if "/" in k)))
            for prefix in prefixes:
                prefix_metrics = {
                    k.split("/", 1)[1]: v
                    for k, v in data.items()
                    if k.startswith(f"{prefix}/")
                }
                print(f"\n  {prefix}:")
                print(f"  {'Metryka':<25} {'Średnia':>10} {'Odch. Std (±)':>14} {'Wynik (Średnia ± Std)':>25}")
                print(f"  {'─' * 76}")
                for metric, val in sorted(prefix_metrics.items()):
                    if isinstance(val, dict):
                        m, s, d = val["mean"], val["std"], val["display"]
                    else:
                        m, s, d = val, 0.0, f"{val:.4f}"
                    print(f"  {metric:<25} {m:>10.4f} {s:>14.4f} {d:>25}")
        else:
            print(f"\n  Metryki:")
            print(f"  {'Metryka':<25} {'Średnia':>10} {'Odch. Std (±)':>14} {'Wynik (Średnia ± Std)':>25}")
            print(f"  {'─' * 76}")
            for metric, val in sorted(data.items()):
                if isinstance(val, dict):
                    m, s, d = val["mean"], val["std"], val["display"]
                else:
                    m, s, d = val, 0.0, f"{val:.4f}"
                print(f"  {metric:<25} {m:>10.4f} {s:>14.4f} {d:>25}")

    print(f"\n{'=' * 100}")
    print(f"Wyniki zapisane do: {args.output}")
    print(f"{'=' * 100}")

    # Generowanie wykresów porównawczych
    generate_comparison_plots(averaged, args.plots_dir)


if __name__ == "__main__":
    main()
