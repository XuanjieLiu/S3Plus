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


class TesterInducedProbing(TesterInduced):
    def __init__(self, config):
        super(TesterInducedProbing, self).__init__(config)

    def prepare_data(self):
        """
        Load the data.
        """
        config = self.config
        dataloader_module = import_module("dataloader." + config["dataloader"])

        self.S_LIST = dataloader_module.S_LIST
        self.C_LIST = dataloader_module.C_LIST

        self.data_dir = config["data_dir"]

        self.train_loader = dataloader_module.get_dataloader(
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            data_type=12,  # use all data for probing
            mode="major",
        )

        self.test_loader = dataloader_module.get_dataloader(
            data_dir=os.path.join(self.data_dir),
            n_segments=12,  # for easier batching
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            test=True,
            mode="major",
        )
        print(len(self.test_loader), "test batches loaded")

    def test(self, **kwargs):  # so that it can accept arbitrary kwargs
        config = self.config

        codebook = self.model.vq.codebook

        self.interval_prober = nn.Linear(self.model.vq.codebook.shape[1] * 2, 12).to(
            self.device
        )  # 12 intervals [-5, 6]

        self.interval_prober = nn.Sequential(
            nn.Linear(self.model.vq.codebook.shape[1] * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
        )
        self.interval_prober.to(self.device)

        optimizer_interval_prober = optim.AdamW(
            self.interval_prober.parameters(), lr=1e-3
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_interval_prober, T_max=50, eta_min=1e-2
        )

        # self.comparison_prober = nn.Linear(self.model.vq.codebook.shape[1] * 2, 1)
        # optimizer_comparison_prober = optim.AdamW(
        #     self.comparison_prober.parameters(), lr=1e-3
        # )

        self.model.eval()
        step = 0
        val_accs = []
        while step < 250:
            # training loop
            self.interval_prober.train()
            batch_data, c_labels, s_labels = next(self.train_loader)
            batch_data = batch_data.to(self.device)
            c_labels = c_labels.to(self.device)
            # forward
            with torch.no_grad():
                zc_vq, indices, commit_loss, zs = self.model.encode(
                    batch_data, quantize=True
                )
            # from c_labels we get C_len^2 pairs of training samples for the probers
            rebatched_input_data = []
            rebatched_gt = []
            for i in range(c_labels.shape[1]):
                for j in range(c_labels.shape[1]):
                    input_data = torch.cat(
                        [zc_vq[:, i], zc_vq[:, j]], dim=-1
                    )  # (B, 2*D)
                    rebatched_input_data.append(input_data)
                    interval_gt = (c_labels[:, j] - c_labels[:, i]) % 12
                    rebatched_gt.append(interval_gt)

            input_data = torch.cat(rebatched_input_data, dim=0)  # (B*C_len^2, 2*D)
            interval_gt = torch.cat(rebatched_gt, dim=0)

            interval_prober_output = self.interval_prober(input_data)
            interval_loss = F.cross_entropy(interval_prober_output, interval_gt)
            optimizer_interval_prober.zero_grad()
            interval_loss.backward()
            optimizer_interval_prober.step()

            print(f"Step {step}, Interval Prober Loss: {interval_loss.item():.4f}")

            # train accuracy
            interval_preds = torch.argmax(interval_prober_output, dim=-1)
            interval_acc = (interval_preds == interval_gt).float().mean().item()
            print(f"Step {step}, Interval Prober Train Acc: {interval_acc:.4f}")

            # validation loop
            if step % 10 == 0:
                for val_batch_data, val_c_labels, val_s_labels in self.test_loader:
                    self.interval_prober.eval()

                    val_batch_data = val_batch_data.to(self.device)
                    val_c_labels = val_c_labels.to(self.device)
                    with torch.no_grad():
                        val_zc_vq, val_indices, val_commit_loss, val_zs = (
                            self.model.encode(val_batch_data, quantize=True)
                        )
                    rebatched_val_input_data = []
                    rebatched_val_gt = []
                    for i in range(val_c_labels.shape[1]):
                        for j in range(val_c_labels.shape[1]):
                            val_input_data = torch.cat(
                                [val_zc_vq[:, i], val_zc_vq[:, j]], dim=-1
                            )  # (B, 2*D)
                            rebatched_val_input_data.append(val_input_data)
                            val_interval_gt = (
                                val_c_labels[:, j] - val_c_labels[:, i]
                            ) % 12
                            rebatched_val_gt.append(val_interval_gt)

                    val_input_data = torch.cat(
                        rebatched_val_input_data, dim=0
                    )  # (B*C_len^2, 2*D)
                    val_interval_gt = torch.cat(rebatched_val_gt, dim=0)

                    val_interval_prober_output = self.interval_prober(val_input_data)
                    val_interval_loss = F.cross_entropy(
                        val_interval_prober_output, val_interval_gt
                    )

                    val_interval_preds = torch.argmax(
                        val_interval_prober_output, dim=-1
                    )
                    val_interval_acc = (
                        (val_interval_preds == val_interval_gt).float().mean().item()
                    )
                    val_accs.append(val_interval_acc)

                    print(
                        f"Step {step}, Interval Prober Val Loss: {val_interval_loss.item():.4f}, Val Acc: {val_interval_acc:.4f}"
                    )

                    assert len(self.test_loader) == 1  # only one validation batch

            step += 1
            scheduler.step()

        print(
            f"Best Interval Prober Val Acc: {max(val_accs):.4f} at step {val_accs.index(max(val_accs)) * 10}"
        )

        self._save_results(
            column_names=["interval_prober_val_acc"],
            results_data=[max(val_accs)],
        )
