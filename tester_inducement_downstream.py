import os
import csv
from tqdm import tqdm
from pathlib import Path
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
import librosa as lr
from soundfile import write
from matplotlib import pyplot as plt
import seaborn as sns
import wandb

from utils.training_utils import *
from utils.eval_utils import *
from tester import Tester
from tester_inducement import TesterInduced


class TesterInducedDownstream(TesterInduced):
    def __init__(self, config):
        super(TesterInducedDownstream, self).__init__(config)

    def prepare_data(self):
        """
        Load the data.
        """
        config = self.config
        dataloader_module = import_module("dataloader." + config["dataloader"])

        self.S_LIST = dataloader_module.S_LIST
        self.C_LIST = dataloader_module.C_LIST

        self.data_dir = config["data_dir"]

        self.test_loader = dataloader_module.get_dataloader(
            data_dir=os.path.join(self.data_dir),
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            test=True,
            mode=config["downstream"],
        )
        print(len(self.test_loader), "test batches loaded")

        self.inducement_loader = dataloader_module.get_dataloader(
            batch_size=config["batch_size"],
            n_segments=6,
            num_workers=config["num_workers"],
            data_type=config["data_type"],
            mode=config["inducement"],
        )
