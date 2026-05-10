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


class TesterInduced(Tester):
    def __init__(self, config):
        super(TesterInduced, self).__init__(config)

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
        )
        print(len(self.test_loader), "test batches loaded")

        self.inducement_loader = dataloader_module.get_dataloader(
            batch_size=config["batch_size"],
            n_segments=6,
            num_workers=config["num_workers"],
            data_type=config["data_type"],
            mode=config["inducement"],
        )

    def test(
        self,
        future_pred_acc=False,
        recon_acc=False,
        future_pred_waveform=False,
        recon_waveform=False,
        vis_tsne=False,
        confusion_mtx=False,
    ):
        """
        testing loop
        """
        config = self.config

        inducement_loader = self.inducement_loader

        self.zc_future_gt = []
        self.zc_future_pred = []
        self.zc_idx_future_gt = []
        self.zc_idx_future_pred = []
        self.c_labels_future_gt = []
        self.x_future_pred = []

        self.x_prompt_gt = []
        self.x_prompt_recon = []

        for i in tqdm(
            range(10), desc="Testing", ncols=100
        ):  # since there is stochasticity in test_loader, we sample 10 times
            for j, batch in enumerate(self.test_loader):
                batch_data, c_labels, s_labels = batch
                batch_data_induced, c_labels_induced, s_labels_induced = next(
                    inducement_loader
                )
                # Move data to device
                batch_data = batch_data.to(device=self.device)
                batch_data_induced = batch_data_induced.to(device=self.device)
                # forward
                with torch.no_grad():
                    losses = self.loss.compute_loss(
                        self.current_step, self.model, batch_data, batch_data_induced
                    )
                    zc_vq, zc_idx, commit_loss, zs = self.model.encode(
                        batch_data, quantize=True
                    )
                    zc_prompt, zc_idx_prompt = (
                        zc_vq[:, :7, :].clone(),
                        zc_idx[:, :7].clone(),
                    )
                    zc_future_pred = self.model.unroll(zc_prompt, 7)
                    zc_future_pred_vq, zc_idx_future_pred, _ = self.model.quantize(
                        zc_future_pred
                    )
                    x_future_pred = self.model.decode(zc_future_pred_vq, zs)
                    x_prompt_recon = self.model.decode(zc_prompt, zs)
                self.zc_future_gt.append(zc_vq[:, 7:14, :].cpu().numpy())
                self.zc_future_pred.append(zc_future_pred_vq.cpu().numpy())
                self.zc_idx_future_gt.append(zc_idx[:, 7:14].cpu().numpy())
                self.zc_idx_future_pred.append(zc_idx_future_pred.cpu().numpy())
                self.c_labels_future_gt.append(c_labels[:, 7:14].cpu().numpy())
                self.x_future_pred.append(x_future_pred.cpu().numpy())
                self.x_prompt_gt.append(batch_data[:, :7, :].cpu().numpy())
                self.x_prompt_recon.append(x_prompt_recon.cpu().numpy())

        self.zc_future_gt = np.concatenate(self.zc_future_gt, axis=0)
        self.zc_future_pred = np.concatenate(self.zc_future_pred, axis=0)
        self.zc_idx_future_gt = np.concatenate(self.zc_idx_future_gt, axis=0)
        self.zc_idx_future_pred = np.concatenate(self.zc_idx_future_pred, axis=0)
        self.c_labels_future_gt = np.concatenate(self.c_labels_future_gt, axis=0)
        self.x_future_pred = np.concatenate(self.x_future_pred, axis=0)
        self.x_prompt_gt = np.concatenate(self.x_prompt_gt, axis=0)
        self.x_prompt_recon = np.concatenate(self.x_prompt_recon, axis=0)

        if future_pred_acc:
            self.future_pred_acc_z()
            self.future_pred_acc_x()
        if recon_acc:
            self.recon_acc_x()

        if recon_waveform:
            self.recon_waveform(n_samples=10)
        if future_pred_waveform:
            self.future_pred_waveform(n_samples=10)
