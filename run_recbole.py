import logging
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import FPMC, GRU4Rec, GRU4RecF, NARM, STAMP, SASRec, SRGNN
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from pathlib import Path
import traceback
import torch
import shutil
import json
import gc
import os

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

model_dict = {
    'FPMC': {
        'parameter_dict': {
                
        },
        'model': FPMC
    },
    'GRU4Rec': {
        'parameter_dict': {
            'train_neg_sample_args': None,
            'neg_sampling': None                        
        },
        'model': GRU4Rec
    },
    'GRU4RecF': {
        'parameter_dict': {
            'train_neg_sample_args': None,  
            'neg_sampling': None,
        },
        'model': GRU4RecF
    },
    'NARM': {
        'parameter_dict': {
            'train_neg_sample_args': None,
            'neg_sampling': None
        },
        'model': NARM
    },
    'STAMP': {
        'parameter_dict': {
            'train_neg_sample_args': None,
            'neg_sampling': None,
        },
        'model': STAMP
    },
    'SASRec': {
        'parameter_dict': {
            'train_neg_sample_args': None,
            'neg_sampling': None
        },
        'model': SASRec
    },
    'SRGNN': {
        'parameter_dict': {
            'train_neg_sample_args': None,
            'neg_sampling': None
        },
        'model': SRGNN
    }
}

dataset_dir = Path("dataset")
dataset_dict = {
    f"{p.name}": f"{p.name}"
    for p in dataset_dir.iterdir()
    if p.is_dir()
}
logger = getLogger()

for dataset_name in dataset_dict.keys():
    if os.path.exists(f"recbole/properties/dataset/{dataset_name}.yaml"):
        os.remove(f"recbole/properties/dataset/{dataset_name}.yaml")

for dataset_name in dataset_dict.keys():
    if not os.path.exists(f"recbole/properties/dataset/{dataset_name}.yaml"):
        base_dataset = dataset_name.split("__")[0]
        template_yaml = f"recbole/properties/dataset/{base_dataset}.yaml"
        if not os.path.exists(template_yaml):
            template_yaml = 'recbole/properties/dataset/30music.yaml'
        shutil.copy(
            template_yaml, 
            f'recbole/properties/dataset/{dataset_name}.yaml'
        )

for model_name in model_dict.keys():
    for dataset_name in dataset_dict.keys():
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            handler.close()

        if os.path.exists(f"saved/{model_name}_{dataset_name}"):
            continue

        config = None
        dataset = None
        train_data = None
        valid_data = None
        test_data = None
        model = None
        trainer = None

        try:
            config = Config(
                model=model_name, 
                dataset=dataset_name, 
                config_dict={
                    **model_dict[model_name]['parameter_dict'],
                    'checkpoint_dir': f'saved/{model_name}_{dataset_name}',
                    'save_dataset': False,
                    'epochs': 100
                }
            )

            init_seed(config['seed'], config['reproducibility'])
            init_logger(config)
            if not logger.handlers:
                c_handler = logging.StreamHandler()
                c_handler.setLevel(logging.INFO)
                logger.addHandler(c_handler)
            
            logger.info(config)
            dataset = create_dataset(config)
            logger.info(dataset)

            train_data, valid_data, test_data = data_preparation(config, dataset)
            model = model_dict[model_name]['model'](config, train_data.dataset).to(config['device'])
            logger.info(model)

            trainer = Trainer(config, model)
            best_valid_score, best_valid_result = trainer.fit(
                train_data,
                valid_data,
                saved=True,
                show_progress=True
            )
            test_result = trainer.evaluate(test_data)

            with open(f'saved/{model_name}_{dataset_name}/results_1.json', 'w') as f:
                json.dump({"test_result": test_result}, f, indent=2)

        except Exception as e:
            logger.error(f"Failed {model_name} on {dataset_name}: {e}")
            traceback.print_exc()
        finally:
            if model is not None:
                try:
                    model.cpu()
                except Exception:
                    pass
            del model, trainer, dataset, train_data, valid_data, test_data, config
            model = None
            trainer = None
            dataset = None
            train_data = None
            valid_data = None
            test_data = None
            config = None
            gc.collect()
            torch.cuda.empty_cache()

for dataset_name in dataset_dict.keys():
    if os.path.exists(f"recbole/properties/dataset/{dataset_name}.yaml"):
        os.remove(f"recbole/properties/dataset/{dataset_name}.yaml")



