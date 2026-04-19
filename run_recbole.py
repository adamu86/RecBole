import logging
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import GRU4Rec, STAMP, SRGNN
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from pathlib import Path
from functools import partial
import torch
import json
import gc

_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load


model_dict = {
    'GRU4Rec': {
        'parameter_dict': {
            'train_neg_sample_args': None,
            'neg_sampling': None
        },
        'model': GRU4Rec
    },
    'STAMP': {
        'parameter_dict': {
            'train_neg_sample_args': None,
            'neg_sampling': None
        },
        'model': STAMP
    }
}

results = {}
dataset_dir = Path("dataset")
dataset_dict = {p.name: p for p in dataset_dir.iterdir() if p.is_dir()}
logger = getLogger()

for model_name in model_dict.keys():
    for dataset_name in dataset_dict.keys():
        try:
            config = Config(
                model=model_name, 
                dataset=dataset_name, 
                config_dict={
                    **model_dict[model_name]['parameter_dict'],
                    'checkpoint_dir': f'saved/{model_name}_{dataset_name}',
                    'save_dataset': False
                }
            )

            init_seed(config['seed'], config['reproducibility'])
            init_logger(config)
            if not logger.handlers:
                c_handler = logging.StreamHandler()
                c_handler.setLevel(logging.INFO)
                logger.addHandler(c_handler)
            
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

            logger.info(f"Best valid score: {best_valid_score}, result: {best_valid_result}")
            logger.info(f"Test result for {model_name} on {dataset_name}: {test_result}")

            results[f"{model_name}_{dataset_name}"] = {
                'best_valid_score': best_valid_score,
                'best_valid_result': best_valid_result,
                'test_result': test_result
            }

            with open('results.json', 'w') as f:
                json.dump(results, f, indent=2)

            del model, trainer, dataset, train_data, valid_data, test_data
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Failed {model_name} on {dataset_name}: {e}")
            results[f"{model_name}_{dataset_name}"] = {'error': str(e)}
            torch.cuda.empty_cache()
            continue
