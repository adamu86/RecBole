import logging
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import GRU4Rec, NARM
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger


parameter_dict = {
    'train_neg_sample_args': None,
    'neg_sampling': None
}

config = Config(model='NARM', dataset='data', config_dict=parameter_dict)


init_seed(config['seed'], config['reproducibility'])


init_logger(config)
logger = getLogger()

c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
logger.addHandler(c_handler)


logger.info(config)

dataset = create_dataset(config)
logger.info(dataset)

train_data, valid_data, test_data = data_preparation(config, dataset)

model = NARM(config, train_data.dataset).to(config['device'])
logger.info(model)


trainer = Trainer(config, model)


best_valid_score, best_valid_result = trainer.fit(
    train_data,
    valid_data,
    show_progress=True
)

print(best_valid_score, best_valid_result)

