"""Re-evaluate saved RecBole models that are missing evaluation results (results_1.json).

Loads saved .pth model checkpoints from saved/ directory for models where results_1.json
does not exist, runs evaluation on test data, saves results_1.json in the exact same format
as run_recbole.py, and prints/saves a summary table.
"""

import os
import glob
import json
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Patch torch.load for PyTorch >= 2.6 compatibility with RecBole checkpoints
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Patch NumPy >= 1.24 compatibility for RecBole (where np.float, np.int, np.bool were removed)
import numpy as np
for _attr, _type in [
    ("float", float),
    ("int", int),
    ("bool", bool),
    ("complex", complex),
    ("object", object),
    ("str", str),
    ("long", int),
    ("unicode", str),
]:
    if not hasattr(np, _attr):
        setattr(np, _attr, _type)

from recbole.quick_start.quick_start import load_data_and_model
from recbole.trainer import Trainer

def find_latest_checkpoint(dir_path):
    """Return the most recently modified .pth file in dir_path."""
    pth_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.pth')]
    if not pth_files:
        return None
    return max(pth_files, key=os.path.getmtime)

def reevaluate_models(saved_base="saved", dataset_filter=None, force=False):
    if not os.path.exists(saved_base):
        print(f"Katalog {saved_base} nie istnieje.")
        return

    # Subdirectories in saved/ (excluding average_results.json and comparison_plots)
    subdirs = [
        os.path.join(saved_base, d) for d in os.listdir(saved_base)
        if os.path.isdir(os.path.join(saved_base, d)) and d != "comparison_plots"
    ]

    print(f"Znaleziono {len(subdirs)} katalogów modeli w '{saved_base}/'.")
    
    all_results = []

    for model_dir in sorted(subdirs):
        dir_name = os.path.basename(model_dir)
        
        if dataset_filter and dataset_filter not in dir_name:
            continue

        results_1_path = os.path.join(model_dir, "results_1.json")
        if os.path.exists(results_1_path) and not force:
            print(f"[SKIP] {dir_name} już posiada results_1.json")
            continue

        ckpt_file = find_latest_checkpoint(model_dir)
        if not ckpt_file:
            print(f"[SKIP] Brak pliku .pth w {dir_name}")
            continue

        print("\n" + "=" * 80)
        print(f" EWALUACJA MODELU: {dir_name}")
        print(f" Plik checkpointu: {os.path.basename(ckpt_file)}")
        print("=" * 80)

        try:
            # Wczytanie konfiguracji, modelu oraz dataloaderów z checkpointu
            config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
                model_file=ckpt_file
            )

            # Inicjalizacja Trenera i ewaluacja na zbiorze testowym
            trainer = Trainer(config, model)
            trainer.eval_collector.data_collect(train_data)
            test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=True)

            # Zapisz wyniki ewaluacji do pliku results_1.json (format spójny z resztą modeli)
            with open(results_1_path, "w", encoding="utf-8") as f:
                json.dump({"test_result": test_result}, f, indent=2)

            print(f"✓ Zapisano wyniki do: {results_1_path}")

            # Wyciągnij kluczowe metryki do tabeli podsumowującej
            row = {
                "Folder": dir_name,
                "Model": config["model"],
                "Dataset": config["dataset"],
                "Recall@10": test_result.get("recall@10", None),
                "MRR@10": test_result.get("mrr@10", None),
                "NDCG@10": test_result.get("ndcg@10", None),
                "Coverage@10": test_result.get("itemcoverage@10", None),
                "AvgPop@10": test_result.get("averagepopularity@10", None),
                "Gini@10": test_result.get("giniindex@10", None),
            }
            all_results.append(row)

            # Czyszczenie pamięci GPU/RAM
            del model, trainer, dataset, train_data, valid_data, test_data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Błąd podczas ewaluacji {dir_name}: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        df = pd.DataFrame(all_results)
        summary_csv = os.path.join(saved_base, "reevaluation_summary.csv")
        df.to_csv(summary_csv, index=False)
        print("\n" + "=" * 80)
        print(" PODSUMOWANIE EWALUACJI")
        print("=" * 80)
        print(df.to_string(index=False))
        print(f"\n✓ Zapisano pełne podsumowanie tabelaryczne do: {summary_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ponowna ewaluacja zapisanych modeli RecBole nieposiadających results_1.json.")
    parser.add_argument("--dataset", type=str, default=None, help="Opcjonalny filtr nazwy zbioru")
    parser.add_argument("--force", action="store_true", help="Wymuś ewaluację nawet jeśli results_1.json istnieje")
    args = parser.parse_args()

    reevaluate_models(dataset_filter=args.dataset, force=args.force)

