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
            # Dodaj prefix do kluczy, żeby odróżnić results_1 od results_N
            prefix = filename.replace(".json", "")  # "results_1" lub "results_N"
            test_result = data.get("test_result", {})
            for metric, value in test_result.items():
                results[f"{prefix}/{metric}"] = value
    return results


def main():
    model_results: dict[str, list[dict]] = defaultdict(list)

    for folder in sorted(SAVED_DIR.iterdir()):
        if not folder.is_dir():
            continue
        model_name = folder.name.split("_")[0]
        results = load_results(folder)
        if results:
            model_results[model_name].append({
                "split": folder.name,
                "metrics": results
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

        averaged[model_name] = {
            "num_splits": len(splits),
            "averaged_metrics": avg_metrics
        }

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


if __name__ == "__main__":
    main()
