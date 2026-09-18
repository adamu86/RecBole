import logging
from logging import getLogger
import json
import os
import glob
import gc
import torch
import numpy as np
import argparse
from recbole.quick_start.quick_start import load_data_and_model
from recbole.trainer import Trainer
from recbole.utils import init_seed

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

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--weight", type=float, default=0.25)
    args = parser.parse_args()

    model_name = "GRU4RecF"
    rerank_field = "artist_tags"
    rerank_topk = args.topk
    rerank_weight = args.weight

    saved_dirs = sorted(glob.glob(f"saved/{model_name}_*"))

    for save_dir in saved_dirs:
        results_json = f"{save_dir}/results_reranking.json"
        
        if os.path.exists(results_json):
            continue

        checkpoint_files = glob.glob(glob.escape(save_dir) + f"/{model_name}-*.pth")

        if not checkpoint_files:
            continue
        
        checkpoint_file = checkpoint_files[0]

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            handler.close()

        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(checkpoint_file)

        config.final_config_dict['rerank_topk'] = None
        init_seed(config['seed'], config['reproducibility'])

        trainer = Trainer(config, model)
        trainer.eval_collector.data_collect(train_data)
        baseline_result = trainer.evaluate(test_data, load_best_model=True, model_file=checkpoint_file, show_progress=True)

        del model, trainer, dataset, train_data, valid_data, test_data
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            handler.close()

        config2, model2, dataset2, train_data2, valid_data2, test_data2 = load_data_and_model(checkpoint_file)

        config2.final_config_dict['rerank_topk'] = rerank_topk
        config2.final_config_dict['rerank_weight'] = rerank_weight
        config2.final_config_dict['rerank_field'] = rerank_field
        init_seed(config2['seed'], config2['reproducibility'])

        trainer2 = Trainer(config2, model2)
        trainer2.eval_collector.data_collect(train_data2)
        rerank_result = trainer2.evaluate(test_data2, load_best_model=True, model_file=checkpoint_file, show_progress=True)

        del model2, trainer2, dataset2, train_data2, valid_data2, test_data2
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        comparison = {
            "baseline": {k: float(v) for k, v in baseline_result.items()},
            "reranking": {k: float(v) for k, v in rerank_result.items()}
        }

        with open(results_json, 'w') as f:
            json.dump(comparison, f, indent=2)

if __name__ == "__main__":
    main()