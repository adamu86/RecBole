"""Test script for tag-based Jaccard reranking.

Loads saved GRU4Rec checkpoints and runs evaluation twice:
1. Without reranking (baseline)
2. With reranking (rerank_topk=100, rerank_weight=1.0)

Runs over all GRU4Rec splits in saved/ and compares metrics side-by-side.
"""
import logging
from logging import getLogger
import json
import os
import glob
import gc
import torch
import numpy as np
import shutil

# Fix for NumPy >= 1.24 where np.float was removed
if not hasattr(np, 'float'):
    np.float = np.float64

from recbole.quick_start.quick_start import load_data_and_model
from recbole.trainer import Trainer
from recbole.utils import init_seed

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# === CONFIGURATION ===
model_name = "GRU4Rec"
saved_dirs = glob.glob(f"saved/{model_name}_*")
saved_dirs.sort()

all_comparisons = {}

for save_dir in saved_dirs:
    dataset_name = os.path.basename(save_dir).replace(f"{model_name}_", "")
    print("\n" + "#"*80)
    print(f"  EVALUATING DATASET: {dataset_name}")
    print("#"*80)
    
    # Find checkpoint
    checkpoint_files = glob.glob(glob.escape(save_dir) + f"/{model_name}-*.pth")
    if not checkpoint_files:
        print(f"No checkpoint found in {save_dir}. Skipping...")
        continue
    checkpoint_file = max(checkpoint_files, key=os.path.getmtime)
    print(f"Using checkpoint: {checkpoint_file}")

    yaml_dst = f"recbole/properties/dataset/{dataset_name}.yaml"
    if not os.path.exists(yaml_dst):
        shutil.copy('recbole/properties/dataset/30music.yaml', yaml_dst)

    try:
        # ============================================
        # 1. BASELINE: Evaluation WITHOUT reranking
        # ============================================
        print("\n  [1/2] BASELINE: Evaluation WITHOUT reranking")
        
        # Suppress some recbole logs for cleaner output
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            handler.close()

        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
            model_file=checkpoint_file
        )

        config.final_config_dict['rerank_topk'] = None
        init_seed(config['seed'], config['reproducibility'])

        trainer = Trainer(config, model)
        trainer.eval_collector.data_collect(train_data)
        baseline_result = trainer.evaluate(test_data, load_best_model=True,
                                           model_file=checkpoint_file, show_progress=True)

        # Clean up
        del model, trainer, dataset, train_data, valid_data, test_data
        gc.collect()
        torch.cuda.empty_cache()

        # ============================================
        # 2. RERANKING: Evaluation WITH reranking
        # ============================================
        print("\n  [2/2] RERANKING: Evaluation WITH Jaccard reranking (topk=100, weight=1.0)")
        
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            handler.close()

        config2, model2, dataset2, train_data2, valid_data2, test_data2 = load_data_and_model(
            model_file=checkpoint_file
        )

        config2.final_config_dict['rerank_topk'] = 100
        config2.final_config_dict['rerank_weight'] = 1.0
        init_seed(config2['seed'], config2['reproducibility'])

        trainer2 = Trainer(config2, model2)
        trainer2.eval_collector.data_collect(train_data2)
        rerank_result = trainer2.evaluate(test_data2, load_best_model=True,
                                          model_file=checkpoint_file, show_progress=True)

        # ============================================
        # 3. COMPARISON
        # ============================================
        print(f"\nRESULTS FOR {dataset_name}:")
        print(f"{'Metric':<25} {'Baseline':>10} {'Reranking':>10} {'Diff':>10} {'Change':>8}")
        print("-" * 65)
        for key in baseline_result:
            base_val = baseline_result[key]
            rerank_val = rerank_result[key]
            diff = rerank_val - base_val
            pct = (diff / base_val * 100) if base_val != 0 else 0
            arrow = "+" if diff > 0 else ("-" if diff < 0 else "=")
            print(f"{key:<25} {base_val:>10.4f} {rerank_val:>10.4f} {diff:>+10.4f} {arrow}{abs(pct):>6.1f}%")

        comparison = {
            "baseline": {k: float(v) for k, v in baseline_result.items()},
            "reranking": {k: float(v) for k, v in rerank_result.items()},
            "config": {"rerank_topk": 100, "rerank_weight": 1.0}
        }
        all_comparisons[dataset_name] = comparison

        with open(f"{save_dir}/rerank_comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)
            
        del model2, trainer2, dataset2, train_data2, valid_data2, test_data2
        gc.collect()
        torch.cuda.empty_cache()

    finally:
        if os.path.exists(yaml_dst):
            os.remove(yaml_dst)

# Final summary file
with open(f"saved/{model_name}_all_rerank_comparisons.json", 'w') as f:
    json.dump(all_comparisons, f, indent=2)
print(f"\nAll results saved to saved/{model_name}_all_rerank_comparisons.json")
