import os
import random
from pathlib import Path

import numpy as np


class Configuration:
    VER = 1
    RAW_DATA_PATH = Path(__file__).parents[1].joinpath("data/raw")
    INTERIM_DATA_PATH = Path(__file__).parents[1].joinpath("data/interim")
    MODEL = Path(__file__).parents[1].joinpath("models")
    target_col = "price"
    seed = 3407
    n_folds = 10
    verbose = 0
    early_stopping_round = 20


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
