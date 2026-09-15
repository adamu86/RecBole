"""Re-evaluate saved RecBole models that are missing evaluation results (results_1.json).

Loads saved .pth model checkpoints from saved/ directory for models where results_1.json
does not exist, runs evaluation on test data, saves results_1.json in the exact same format
as run_recbole.py, and prints/saves a summary table.
"""

import os
import gc
import glob
import json
import logging
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Patch torch.load for PyTorch >= 2.6 compatibility with RecBole checkpoints
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    if not torch.cuda.is_available():
        kwargs.setdefault('map_location', 'cpu')
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

def extract_checkpoint_info(ckpt_file):
    """Odczytaj informacje o epoce i walidacji bezpośrednio z pliku .pth checkpointu."""
    try:
        ckpt_data = torch.load(ckpt_file, map_location='cpu')
        raw_epoch = ckpt_data.get('epoch', None)
        # W RecBole 'epoch' to indeks epoki (0-indexed), w której model uzyskał najlepszy wynik walidacji
        best_epoch = (int(raw_epoch) + 1) if raw_epoch is not None else None

        best_valid_score = ckpt_data.get('best_valid_score', None)
        if hasattr(best_valid_score, 'item'):
            best_valid_score = best_valid_score.item()
        elif best_valid_score is not None:
            try:
                best_valid_score = float(best_valid_score)
            except Exception:
                pass

        ckpt_cfg = ckpt_data.get('config', {})
        stopping_step = None
        max_epochs = None
        model_name = None
        dataset_name = None

        if hasattr(ckpt_cfg, '__getitem__'):
            try:
                stopping_step = ckpt_cfg['stopping_step']
            except Exception:
                pass
            try:
                max_epochs = ckpt_cfg['epochs']
            except Exception:
                pass
            try:
                model_name = ckpt_cfg['model']
            except Exception:
                pass
            try:
                dataset_name = ckpt_cfg['dataset']
            except Exception:
                pass

        # Oszacowanie epoki zatrzymania na podstawie patience / early stopping z konfiguracji wewnątrz .pth
        if best_epoch is not None and stopping_step is not None:
            stopped_epoch_est = best_epoch + int(stopping_step)
            if max_epochs is not None:
                stopped_epoch_est = min(stopped_epoch_est, int(max_epochs))
        else:
            stopped_epoch_est = None

        del ckpt_data
        return {
            'best_epoch': best_epoch,
            'stopped_epoch_est': stopped_epoch_est,
            'best_valid_score': best_valid_score,
            'model': model_name,
            'dataset': dataset_name,
        }
    except Exception as e:
        print(f"Błąd podczas odczytu checkpointu {ckpt_file}: {e}")
        return {
            'best_epoch': None,
            'stopped_epoch_est': None,
            'best_valid_score': None,
            'model': None,
            'dataset': None,
        }

def save_epoch_summary(epoch_dict, epoch_json_path):
    """Zapisz podsumowanie epok do pliku JSON oraz CSV."""
    with open(epoch_json_path, "w", encoding="utf-8") as f:
        json.dump(epoch_dict, f, indent=2, ensure_ascii=False)

    csv_path = str(Path(epoch_json_path).with_suffix(".csv"))
    rows = []
    for dir_name, info in epoch_dict.items():
        rows.append({
            "Folder": dir_name,
            "Model": info.get("model"),
            "Dataset": info.get("dataset"),
            "Best_Epoch": info.get("best_epoch"),
            "Stopped_Epoch_Est": info.get("stopped_epoch_est"),
            "Best_Valid_Score": info.get("best_valid_score"),
        })
    if rows:
        pd.DataFrame(rows).to_csv(csv_path, index=False)

def reevaluate_models(saved_base="saved", dataset_filter=None, force=False, skip_models=None, extract_epochs_only=False):
    if not os.path.exists(saved_base):
        print(f"Katalog {saved_base} nie istnieje.")
        return

    # Subdirectories in saved/ (excluding average_results.json and comparison_plots)
    subdirs = [
        os.path.join(saved_base, d) for d in os.listdir(saved_base)
        if os.path.isdir(os.path.join(saved_base, d)) and d != "comparison_plots"
    ]

    print(f"Znaleziono {len(subdirs)} katalogów modeli w '{saved_base}/'.")
    
    summary_csv = os.path.join(saved_base, "reevaluation_summary.csv")
    if os.path.exists(summary_csv):
        all_results = pd.read_csv(summary_csv).to_dict('records')
    else:
        all_results = []
    epoch_json_path = os.path.join(saved_base, "epoch_summary.json")
    if os.path.exists(epoch_json_path):
        with open(epoch_json_path, "r", encoding="utf-8") as f:
            epoch_dict = json.load(f)
    else:
        epoch_dict = {}

    for model_dir in sorted(subdirs):
        dir_name = os.path.basename(model_dir)
        
        if dataset_filter and dataset_filter not in dir_name:
            continue

        if skip_models and any(sm in dir_name for sm in skip_models):
            print(f"[SKIP] {dir_name} — model na liście pomijanych")
            continue

        ckpt_file = find_latest_checkpoint(model_dir)
        if not ckpt_file:
            print(f"[SKIP] Brak pliku .pth w {dir_name}")
            continue

        # Zawsze odczytujemy informacje o epoce bezpośrednio z pliku .pth
        epoch_info = extract_checkpoint_info(ckpt_file)
        epoch_dict[dir_name] = epoch_info
        save_epoch_summary(epoch_dict, epoch_json_path)

        results_1_path = os.path.join(model_dir, "results_1.json")
        if os.path.exists(results_1_path) and not force:
            print(f"[SKIP] {dir_name} już posiada results_1.json (Najlepsza epoka: {epoch_info['best_epoch']})")
            continue

        if extract_epochs_only:
            print(f"[EPOCH] {dir_name}: Najlepsza epoka = {epoch_info['best_epoch']}, Oszacowany stop = {epoch_info['stopped_epoch_est']}, Best Valid Score = {epoch_info['best_valid_score']}")
            continue

        print("\n" + "=" * 80)
        print(f" EWALUACJA MODELU: {dir_name}")
        print(f" Plik checkpointu: {os.path.basename(ckpt_file)}")
        print(f" Najlepsza epoka: {epoch_info['best_epoch']} (oszacowane zatrzymanie: ~{epoch_info['stopped_epoch_est']})")
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
                "Best_Epoch": epoch_info.get("best_epoch"),
                "Stopped_Epoch_Est": epoch_info.get("stopped_epoch_est"),
                "Best_Valid_Score": epoch_info.get("best_valid_score"),
                "Recall@10": test_result.get("recall@10", None),
                "MRR@10": test_result.get("mrr@10", None),
                "NDCG@10": test_result.get("ndcg@10", None),
                "Coverage@10": test_result.get("itemcoverage@10", None),
                "AvgPop@10": test_result.get("averagepopularity@10", None),
                "Gini@10": test_result.get("giniindex@10", None),
            }
            all_results.append(row)

            # Zapisuj CSV natychmiast po każdym modelu
            pd.DataFrame(all_results).to_csv(summary_csv, index=False)

            # Czyszczenie pamięci GPU/RAM
            del config, model, trainer, dataset, train_data, valid_data, test_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Wyczyść handlery logowania (RecBole dodaje nowe za każdym razem)
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
                handler.close()

        except Exception as e:
            print(f"❌ Błąd podczas ewaluacji {dir_name}: {e}")
            import traceback
            traceback.print_exc()


    if extract_epochs_only:
        print("\n" + "=" * 80)
        print(" PODSUMOWANIE EPOK (Z PLIKÓW .PTH)")
        print("=" * 80)
        df_epochs = pd.DataFrame(list(epoch_dict.values()), index=list(epoch_dict.keys()))
        print(df_epochs.to_string())
        print(f"\n✓ Wyniki zapisane w: {epoch_json_path} oraz {Path(epoch_json_path).with_suffix('.csv')}")
    elif all_results:
        df = pd.DataFrame(all_results)
        print("\n" + "=" * 80)
        print(" PODSUMOWANIE EWALUACJI")
        print("=" * 80)
        print(df.to_string(index=False))
        print(f"\n✓ Wyniki zapisane w: {summary_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ponowna ewaluacja zapisanych modeli RecBole nieposiadających results_1.json.")
    parser.add_argument("--dataset", type=str, default=None, help="Opcjonalny filtr nazwy zbioru")
    parser.add_argument("--skip-model", type=str, nargs="+", default=None, help="Nazwy modeli do pominięcia (np. STAMP SASRec)")
    parser.add_argument("--force", action="store_true", help="Wymuś ewaluację nawet jeśli results_1.json istnieje")
    parser.add_argument("--extract-epochs-only", action="store_true", help="Tylko odczytaj epoki i metryki walidacji z plików .pth bez ponownej ewaluacji")
    args = parser.parse_args()

    reevaluate_models(
        dataset_filter=args.dataset,
        force=args.force,
        skip_models=args.skip_model,
        extract_epochs_only=args.extract_epochs_only,
    )