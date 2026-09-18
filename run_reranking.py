import logging
from logging import getLogger
import json
import os
import glob
import gc
import torch
import numpy as np
import shutil
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
    parser.add_argument("--model", type=str, default="GRU4RecF")
    parser.add_argument("--field", type=str, default="artist_tags")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--weight", type=float, default=0.25)
    parser.add_argument("--filter", type=str, default=None)
    args = parser.parse_args()

    model_name = args.model
    rerank_field = args.field
    rerank_topk = args.topk
    rerank_weight = args.weight
    ds_filter = args.filter

    saved_dirs = glob.glob(f"saved/{model_name}_*")
    if ds_filter:
        saved_dirs = [d for d in saved_dirs if ds_filter in d]
    saved_dirs.sort()

    if not saved_dirs:
        return

    for save_dir in saved_dirs:
        dataset_name = os.path.basename(save_dir).replace(f"{model_name}_", "")
        out_json = f"{save_dir}/rerank_{rerank_field}_comparison.json"
        
        if os.path.exists(out_json):
            continue

        checkpoint_files = glob.glob(glob.escape(save_dir) + f"/{model_name}-*.pth")
        if not checkpoint_files:
            continue
        checkpoint_file = max(checkpoint_files, key=os.path.getmtime)

        yaml_dst = f"recbole/properties/dataset/{dataset_name}.yaml"
        if not os.path.exists(yaml_dst):
            template_yaml = 'recbole/properties/dataset/lastfm1k.yaml'
            if os.path.exists(template_yaml):
                shutil.copy(template_yaml, yaml_dst)

        try:
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

            out_json = f"{save_dir}/rerank_{rerank_field}_comparison.json"
            with open(out_json, 'w') as f:
                json.dump(comparison, f, indent=2)
            
        finally:
            if os.path.exists(yaml_dst):
                try:
                    os.remove(yaml_dst)
                except OSError:
                    pass

    all_comparisons = {}
    for save_dir in saved_dirs:
        ds_name = os.path.basename(save_dir).replace(f"{model_name}_", "")
        res_json = f"{save_dir}/rerank_{rerank_field}_comparison.json"
        if os.path.exists(res_json):
            try:
                with open(res_json, 'r') as f:
                    all_comparisons[ds_name] = json.load(f)
            except Exception as e:
                pass

    out_summary = f"saved/{model_name}_all_rerank_{rerank_field}_comparisons.json"
    with open(out_summary, 'w') as f:
        json.dump(all_comparisons, f, indent=2)

if __name__ == "__main__":
    main()