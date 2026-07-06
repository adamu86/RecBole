import logging
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import FPMC, GRU4Rec, GRU4RecMod, NARM, STAMP, SASRec, SRGNN
from recbole.quick_start.quick_start import load_data_and_model
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from evaluate_playlist import evaluate_playlist
from pathlib import Path
import traceback
import torch
import shutil
import json
import gc
import os
import glob

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

model_dict = {
    # 'FPMC': {
    #     'parameter_dict': {
    #         'train_batch_size': 4096,
    #     },
    #     'model': FPMC
    # },
    # 'GRU4Rec': {
    #     'parameter_dict': {
    #         'train_neg_sample_args': None,
    #         'neg_sampling': None
    #     },
    #     'model': GRU4Rec
    # },
    'GRU4RecMod': {
        'parameter_dict': {
            'train_neg_sample_args': None,
            'neg_sampling': None
        },
        'model': GRU4RecMod
    },
    # 'NARM': {
    #     'parameter_dict': {
    #         'train_neg_sample_args': None,
    #         'neg_sampling': None
    #     },
    #     'model': NARM
    # },
    # 'STAMP': {
    #     'parameter_dict': {
    #         'train_neg_sample_args': None,
    #         'neg_sampling': None,
    #     },
    #     'model': STAMP
    # },
    # 'SASRec': {
    #     'parameter_dict': {
    #         'train_neg_sample_args': None,
    #         'neg_sampling': None

    #     },
    #     'model': SASRec
    # },
    # 'SRGNN': {
    #     'parameter_dict': {
    #         'train_neg_sample_args': None,
    #         'neg_sampling': None,
    #         'train_batch_size': 4096
    #     },
    #     'model': SRGNN
    # }
}

dataset_dir = Path("dataset")
dataset_dict = {
    f"M{p.name}": f"M{p.name}"
    for p in dataset_dir.iterdir()
    if p.is_dir()
}
logger = getLogger()

for dataset_name in dataset_dict.keys():
    if os.path.exists(f"recbole/properties/dataset/{dataset_name}.yaml"):
        os.remove(f"recbole/properties/dataset/{dataset_name}.yaml")

for dataset_name in dataset_dict.keys():
    if not os.path.exists(f"recbole/properties/dataset/{dataset_name}.yaml"):
        shutil.copy(
            'recbole/properties/dataset/30music.yaml', 
            f'recbole/properties/dataset/{dataset_name}.yaml'
        )

for model_name in model_dict.keys():
    for dataset_name in dataset_dict.keys():
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            handler.close()

        if os.path.exists(f"saved/{model_name}_{dataset_name}"):
            continue

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

            # evaluate_playlist(
            #     config=config,
            #     model=model,
            #     dataset=dataset,
            #     train_data=train_data,
            #     test_data=test_data
            # )

            del model, trainer, dataset, train_data, valid_data, test_data
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Failed {model_name} on {dataset_name}: {e}")
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue

# dataset_dict = dict(reversed(list(dataset_dict.items())))

# # dokręcanie śruby
# for model_name in model_dict.keys():
#     for dataset_name in dataset_dict.keys():
#         if "days[245-185]" not in dataset_name:
#             continue
#         checkpoint_dir = f"saved/{model_name}_{dataset_name}"
#         if not os.path.exists(checkpoint_dir):
#             continue

#         for handler in logging.root.handlers[:]:
#             logging.root.removeHandler(handler)
#             handler.close()

#         try:
#             config = Config(
#                 model=model_name,
#                 dataset=dataset_name,
#                 config_dict={
#                     **model_dict[model_name]['parameter_dict'],
#                     'checkpoint_dir': checkpoint_dir,
#                     'epochs': 20,
#                     'save_dataset': False
#                 }
#             )

#             init_seed(config['seed'], config['reproducibility'])
#             init_logger(config)
#             if not logger.handlers:
#                 logger.addHandler(logging.StreamHandler())

#             dataset = create_dataset(config)
#             train_data, valid_data, test_data = data_preparation(config, dataset)
#             model = model_dict[model_name]['model'](config, train_data.dataset).to(config['device'])

#             trainer = Trainer(config, model)
            
#             checkpoints = list(Path(checkpoint_dir).glob("*.pth"))
#             latest = max(checkpoints, key=os.path.getmtime)
#             trainer.resume_checkpoint(latest)

#             best_valid_score, best_valid_result = trainer.fit(
#                 train_data, valid_data, saved=True, show_progress=True
#             )
#             test_result = trainer.evaluate(test_data)

#             with open(f'{checkpoint_dir}/results_1.json', 'w') as f:
#                 json.dump({"test_result": test_result}, f, indent=2)

#             evaluate_playlist(
#                 config=config,
#                 model=model,
#                 dataset=dataset,
#                 train_data=train_data,
#                 test_data=test_data
#             )

#             del model, trainer, dataset, train_data, valid_data, test_data
#             gc.collect()
#             torch.cuda.empty_cache()

#         except Exception as e:
#             logger.error(f"Finetune failed {model_name} on {dataset_name}: {e}")
#             traceback.print_exc()

# # sama ewaluacja playlist
# for model_name in model_dict.keys():
#     for dataset_name in dataset_dict.keys():
#         for handler in logging.root.handlers[:]:
#             logging.root.removeHandler(handler)
#             handler.close()

#         save_dir = f"saved/{model_name}_{dataset_name}"
#         if not os.path.exists(save_dir):
#             logger.warning(f"Brak folderu {save_dir}, pomijam")
#             continue

#         checkpoint_files = glob.glob(glob.escape(save_dir) + f"/{model_name}-*.pth")
#         if not checkpoint_files:
#             logger.warning(f"Brak checkpointu w {save_dir}, pomijam")
#             continue

#         checkpoint_file = max(checkpoint_files, key=os.path.getmtime)

#         try:
#             config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
#                 model_file=checkpoint_file
#             )

#             evaluate_playlist(
#                 config=config,
#                 model=model,
#                 dataset=dataset,
#                 train_data=train_data,
#                 test_data=test_data
#             )

#             del model, dataset, train_data, valid_data, test_data
#             gc.collect()
#             torch.cuda.empty_cache()
#         except Exception as e:
#             logger.error(f"Failed {model_name} on {dataset_name}: {e}")
#             traceback.print_exc()
#             torch.cuda.empty_cache()
#             continue

for dataset_name in dataset_dict.keys():
    os.remove(f"recbole/properties/dataset/{dataset_name}.yaml")
