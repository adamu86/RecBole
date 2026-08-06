"""
Test: czy RecBole full_sort_predict zwraca ujemne wartości score?

Cel:
  Empiryczna weryfikacja obawy Clauda, że dot-product scores w modelach
  sekwencyjnych (SASRec, GRU4Rec, itp.) mogą być ujemne — co powoduje
  rozbieżność między kodem (abs) a zamierzonym wzorem matematycznym.

Uruchomienie:
  python test_negative_scores.py [--model GRU4RecF]
"""

import sys
import os
import glob
import shutil
import logging
import argparse

# Fix dla NumPy >= 1.24
import numpy as np
for _attr, _type in [
    ("float", float), ("int", int), ("bool", bool),
    ("complex", complex), ("object", object), ("str", str),
    ("long", int), ("unicode", str),
]:
    if not hasattr(np, _attr):
        setattr(np, _attr, _type)

import torch

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, init_seed, init_logger
from recbole.trainer import Trainer


def analyze_scores_for_model(checkpoint_file, target_field='artist_tags'):
    """Ładuje model i analizuje rozkład surowych score'ów."""
    checkpoint = torch.load(checkpoint_file, weights_only=False)
    config = checkpoint["config"]

    cfg_dict = config.final_config_dict if hasattr(config, 'final_config_dict') else config

    # Remapping pola tagów (jeśli potrzebne)
    if 'selected_features' in cfg_dict:
        feats = cfg_dict['selected_features']
        if 'item_tags' in feats:
            cfg_dict['selected_features'] = [target_field if f == 'item_tags' else f for f in feats]
    if 'load_col' in cfg_dict and isinstance(cfg_dict['load_col'], dict) and 'item' in cfg_dict['load_col']:
        item_cols = cfg_dict['load_col']['item']
        if 'item_tags' in item_cols:
            cfg_dict['load_col']['item'] = [target_field if f == 'item_tags' else f for f in item_cols]
    if 'state_dict' in checkpoint:
        new_state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            new_k = k.replace('.item_tags.', f'.{target_field}.')
            new_state_dict[new_k] = v
        checkpoint['state_dict'] = new_state_dict

    # Wyłącz reranking podczas tego testu
    cfg_dict['rerank_topk'] = None

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    if checkpoint.get("other_parameter") is not None:
        model.load_other_parameter(checkpoint.get("other_parameter"))

    model.eval()
    device = config["device"]

    print(f"\n{'='*70}")
    print(f"  MODEL: {config['model']}")
    print(f"  CHECKPOINT: {os.path.basename(checkpoint_file)}")
    print(f"{'='*70}")

    all_mins = []
    all_maxs = []
    all_means = []
    n_batches = 0
    n_negative_total = 0
    n_total_scores = 0
    first_batch_stats = None

    with torch.no_grad():
        for batch_idx, batched_data in enumerate(test_data):
            if batch_idx >= 20:  # Ogranicz do 20 batchy dla szybkości
                break

            try:
                interaction, history_index, positive_u, positive_i = batched_data
                interaction = interaction.to(device)

                # Wywołaj full_sort_predict (tak jak robi to Trainer)
                if hasattr(model, 'full_sort_predict'):
                    scores = model.full_sort_predict(interaction)
                else:
                    # Fallback: predict na wszystkich itemach
                    n_items = dataset.item_num
                    item_tensor = torch.arange(n_items, device=device)
                    scores = model.predict(interaction.repeat(n_items, 1), item_tensor)
                    scores = scores.view(-1, n_items)

                # Flatten jeśli to nie jest 2D
                if scores.dim() == 1:
                    scores = scores.unsqueeze(0)

                batch_min = scores.min().item()
                batch_max = scores.max().item()
                batch_mean = scores.mean().item()
                n_neg = (scores < 0).sum().item()
                n_total = scores.numel()

                all_mins.append(batch_min)
                all_maxs.append(batch_max)
                all_means.append(batch_mean)
                n_negative_total += n_neg
                n_total_scores += n_total
                n_batches += 1

                if first_batch_stats is None:
                    first_batch_stats = {
                        'min': batch_min,
                        'max': batch_max,
                        'mean': batch_mean,
                        'std': scores.std().item(),
                        'pct_negative': 100.0 * n_neg / n_total,
                        'shape': tuple(scores.shape),
                    }

                if batch_idx < 3:
                    print(f"\n  Batch {batch_idx}: shape={tuple(scores.shape)}")
                    print(f"    min={batch_min:.4f}  max={batch_max:.4f}  mean={batch_mean:.4f}")
                    print(f"    negative: {n_neg}/{n_total} ({100.0*n_neg/n_total:.1f}%)")
                    # Pokaż histogram mini
                    sample = scores.flatten().cpu()
                    neg_count = (sample < 0).sum().item()
                    zero_count = (sample == 0).sum().item()
                    pos_count = (sample > 0).sum().item()
                    print(f"    rozkład: <0: {neg_count}  =0: {zero_count}  >0: {pos_count}")

            except Exception as e:
                print(f"  Błąd w batch {batch_idx}: {e}")
                continue

    if n_batches == 0:
        print("  BRAK BATCHY DO ANALIZY!")
        return None

    # Podsumowanie zbiorcze
    global_min = min(all_mins)
    global_max = max(all_maxs)
    global_mean = np.mean(all_means)
    pct_neg_global = 100.0 * n_negative_total / n_total_scores

    print(f"\n{'─'*70}")
    print(f"  PODSUMOWANIE ({n_batches} batchy, {n_total_scores:,} score'ów łącznie):")
    print(f"{'─'*70}")
    print(f"  Globalny min:    {global_min:.6f}")
    print(f"  Globalny max:    {global_max:.6f}")
    print(f"  Średnia:         {global_mean:.6f}")
    print(f"  % ujemnych:      {pct_neg_global:.2f}%")
    print(f"  Łączna liczba ujemnych: {n_negative_total:,}")
    print()

    # WERDYKT
    if global_min < 0:
        print(f"  ✗ WERDYKT: SCORE'Y MOGĄ BYĆ UJEMNE!")
        print(f"    → Najniższy zaobserwowany score: {global_min:.6f}")
        print(f"    → Obawa Clauda jest UZASADNIONA")
        print(f"    → Wzór 'abs' i 'bez abs' dają RÓŻNE wyniki dla ujemnych score'ów")
        print()
        print(f"  KONSEKWENCJA dla rerankingu:")
        print(f"    KOD (z abs):  score + |score| * α * Jaccard")
        print(f"                  {global_min:.4f} + {abs(global_min):.4f} * α * Jac")
        example_jac = 0.3
        example_alpha = 1.0
        abs_formula = global_min + abs(global_min) * example_alpha * example_jac
        mul_formula = global_min + global_min * example_alpha * example_jac
        print(f"                  dla α=1, Jac={example_jac}: {abs_formula:.6f}")
        print(f"    WZÓR (bez abs): score + score * α * Jaccard")
        print(f"                  {global_min:.4f} + {global_min:.4f} * α * Jac")
        print(f"                  dla α=1, Jac={example_jac}: {mul_formula:.6f}")
        diff = abs_formula - mul_formula
        print(f"    RÓŻNICA:       {diff:.6f}  (kara → bonus lub odwrotnie!)")
    else:
        print(f"  ✓ WERDYKT: Wszystkie score'y są nieujemne w tym eksperymencie.")
        print(f"    → Obawa Clauda jest NIEZASADNA dla tego modelu/datasetu")
        print(f"    → Kod z 'abs' i bez 'abs' dają identyczne wyniki")

    return {
        'min': global_min,
        'max': global_max,
        'mean': global_mean,
        'pct_negative': pct_neg_global,
        'n_negative': n_negative_total,
        'n_total': n_total_scores,
        'has_negative': global_min < 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Test: czy RecBole zwraca ujemne score'y?")
    parser.add_argument("--model", type=str, default="GRU4RecF",
                        help="Nazwa modelu (domyślnie: GRU4RecF)")
    parser.add_argument("--field", type=str, default="artist_tags",
                        help="Pole tagów (domyślnie: artist_tags)")
    args = parser.parse_args()

    model_name = args.model
    saved_dirs = glob.glob(f"saved/{model_name}_*")
    saved_dirs.sort()

    if not saved_dirs:
        print(f"Brak checkpointów saved/{model_name}_*")
        print("Dostępne katalogi w saved/:")
        for d in glob.glob("saved/*"):
            print(f"  {d}")
        return

    print(f"\nZnaleziono {len(saved_dirs)} katalog(ów) dla modelu {model_name}:")
    for d in saved_dirs:
        print(f"  {d}")

    results_summary = {}

    for save_dir in saved_dirs:
        dataset_name = os.path.basename(save_dir).replace(f"{model_name}_", "")
        checkpoint_files = glob.glob(glob.escape(save_dir) + f"/{model_name}-*.pth")
        if not checkpoint_files:
            print(f"\nBrak checkpointu w {save_dir}. Pomijam.")
            continue
        checkpoint_file = max(checkpoint_files, key=os.path.getmtime)

        yaml_dst = f"recbole/properties/dataset/{dataset_name}.yaml"
        created_yaml = False
        if not os.path.exists(yaml_dst):
            shutil.copy('recbole/properties/dataset/30music.yaml', yaml_dst)
            created_yaml = True

        try:
            # Zamknij wszystkie handlery logowania
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
                handler.close()

            result = analyze_scores_for_model(checkpoint_file, target_field=args.field)
            if result:
                results_summary[dataset_name] = result
        except Exception as e:
            print(f"\nBłąd dla {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if created_yaml and os.path.exists(yaml_dst):
                os.remove(yaml_dst)

    # Finalne podsumowanie
    if results_summary:
        print(f"\n{'='*70}")
        print(f"  FINALNE PODSUMOWANIE DLA WSZYSTKICH DATASETÓW")
        print(f"{'='*70}")
        any_negative = False
        for ds, r in results_summary.items():
            flag = "✗ UJEMNE" if r['has_negative'] else "✓ OK"
            print(f"  [{flag}]  {ds:30s}  min={r['min']:.4f}  % neg={r['pct_negative']:.1f}%")
            if r['has_negative']:
                any_negative = True
        print()
        if any_negative:
            print("  → OBAWA CLAUDA UZASADNIONA: przynajmniej jeden dataset ma ujemne score'y")
            print("  → Rozważ zmianę: 'topk_scores.abs() * boosts_t'  →  'topk_scores * boosts_t'")
            print("    (jeśli chcesz, żeby kod dokładnie odpowiadał wzorowi score·(1+α·Jac))")
        else:
            print("  → Obawa Clauda NIEZASADNA dla testowanych modeli/datasetów")
            print("  → Kod z abs() daje te same wyniki co wzór matematyczny")


if __name__ == "__main__":
    main()
