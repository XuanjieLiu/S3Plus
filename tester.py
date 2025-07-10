import os
import gc
import sys
import datetime
import logging
import torch.distributed
import itertools
import yaml
from tqdm import tqdm
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


class Tester:
    def __init__(self, config):
        # basic configs
        self.config = config
        if self.config["debug"]:
            self.portion = 1
            self.config["n_epochs"] = 1
            self.config["n_steps"] = 100
            self.config["log_every_n_steps"] = 1
            self.config["save_top_k"] = 1
        else:
            self.portion = 1

        self.output_dir = os.path.dirname(config["active_checkpoint"])

        # device
        if self.config["device"] == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            # self.local_rank = int(os.environ["LOCAL_RANK"])
        else:
            self.device = torch.device("cpu")
            # self.local_rank = 0

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

    def build_model(self):
        """
        Set up model & optimizer & loss functions.
        Load previous model if specified.
        """
        config = self.config
        method_specs = self.config["method"].split("_")
        self.method_specs = method_specs

        if "ISymm" in method_specs:
            if "insnotes" in config["dataloader"]:
                Model = import_module("model.simple_rnn_insnotes").SymmCSAEwithPrior
                Loss = import_module("model.symm_loss").SymmLoss

        model_config = self.config["model_config"]
        loss_config = self.config["loss_config"]

        # model
        self.model = Model(model_config).to(self.device)

        cp_state_dict = torch.load(config["active_checkpoint"])["model"]
        self.current_step = torch.load(config["active_checkpoint"])["step"]

        self.model.load_state_dict(cp_state_dict, strict=False)
        self.model.eval()

        # loss function
        self.loss = Loss(loss_config, use_isymm="ISymm" in method_specs)

        print("Testing checkpoint ", config["active_checkpoint"], "\n")

    def test(
        self,
        future_pred_acc=False,
        recon_waveform=True,
        future_pred_waveform=True,
        vis_tsne=False,
        confusion_mtx=False,
    ):
        """
        testing loop
        """
        config = self.config

        self.zc_future_gt = []  #
        self.zc_future_pred = []
        self.c_labels_future_gt = []
        self.x_future_pred = []

        self.x_prompt_gt = []
        self.x_prompt_recon = []

        for i, batch in tqdm(enumerate(self.test_loader)):
            batch_data, c_labels, s_labels = batch
            # Move data to device
            batch_data = batch_data.to(device=self.device)
            # forward
            with torch.no_grad():
                losses = self.loss.compute_loss(
                    self.current_step, self.model, batch_data
                )
                zc_vq, indices, commit_loss, zs = self.model.encode(
                    batch_data, quantize=True
                )
                zc_prompt = zc_vq[
                    :, :7, :
                ].clone()  # use the first 12 tokens as prompt, 12 is hard coded
                zc_future_pred = self.model.unroll(zc_prompt, 7)
                zc_future_pred_vq = self.model.quantize(zc_future_pred)[0]
                x_future_pred = self.model.decode(zc_future_pred_vq, zs)
                x_prompt_recon = self.model.decode(zc_prompt, zs)
            self.zc_future_pred.append(zc_future_pred_vq.cpu().numpy())
            self.zc_future_gt.append(zc_vq[:, 7:14, :].cpu().numpy())
            self.c_labels_future_gt.append(c_labels[:, 7:14].cpu().numpy())
            self.x_future_pred.append(x_future_pred.cpu().numpy())
            self.x_prompt_gt.append(batch_data[:, :7, :].cpu().numpy())
            self.x_prompt_recon.append(x_prompt_recon.cpu().numpy())

        self.zc_future_pred = np.concatenate(self.zc_future_pred, axis=0)
        self.zc_future_gt = np.concatenate(self.zc_future_gt, axis=0)
        self.c_labels_future_gt = np.concatenate(self.c_labels_future_gt, axis=0)
        self.x_future_pred = np.concatenate(self.x_future_pred, axis=0)
        self.x_prompt_gt = np.concatenate(self.x_prompt_gt, axis=0)
        self.x_prompt_recon = np.concatenate(self.x_prompt_recon, axis=0)

        if future_pred_acc:
            self.future_pred_acc_z()
            # self.future_pred_acc_x()

        if recon_waveform:
            self.recon_waveform(n_samples=10)
        if future_pred_waveform:
            self.future_pred_waveform(n_samples=10)

    def future_pred_acc_z(self):
        """
        Compute the future prediction accuracy on the zc level (representation level).
        """
        zc_future_pred = self.zc_future_pred
        zc_future_gt = self.zc_future_gt

        # compute the accuracies
        acc = []
        for i in range(zc_future_pred.shape[1]):
            acc.append(
                np.mean(
                    [
                        np.allclose(
                            zc_future_pred[j, i, :],
                            zc_future_gt[j, i, :],
                            atol=1e-3,
                            rtol=1e-3,
                        )
                        for j in range(zc_future_pred.shape[0])
                    ],
                )
            )

        print(f"Future prediction accuracies on Z: {acc}")

    def future_pred_acc_x(self, save_n_samples=10):
        """
        Compute the future prediction accuracy on the x level (behavioral level).
        if save_n_samples > 0, save these audio files of the predictions.
        """
        c_labels_future_gt = self.c_labels_future_gt
        x_future_pred = self.x_future_pred

        config = self.config
        if config["dataloader"] == "insnotes_dataloader":
            sr = 16000
            hop_length = 256
            win_length = 1024
            n_fft = 1024

        acc = []
        n_saved_examples = 0
        for i in range(x_future_pred.shape[1]):
            waveforms = [
                lr.feature.inverse.mel_to_audio(
                    x_i,
                    sr=sr,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                )
                for x_i in x_future_pred[:, i, :]
            ]

        pass
        # TODO

    def recon_waveform(self, n_samples=10):
        """
        Save some x-level inputs and reconstructions for checking.
        """
        x_prompt_gt = self.x_prompt_gt
        x_prompt_recon = self.x_prompt_recon

        config = self.config
        if config["dataloader"] == "insnotes_dataloader":
            sr = 16000
            hop_length = 256
            win_length = 1024
            n_fft = 1024
            # remove the added channel dimension
            x_prompt_gt = x_prompt_gt.squeeze()
            x_prompt_recon = x_prompt_recon.squeeze()

        for i in range(n_samples):
            waveform_gt = []
            waveform_recon = []
            for j in range(x_prompt_gt[i].shape[0]):
                waveform_gt.append(
                    lr.feature.inverse.mel_to_audio(
                        x_prompt_gt[i][j, :],
                        sr=sr,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        win_length=win_length,
                    )
                )
                waveform_recon.append(
                    lr.feature.inverse.mel_to_audio(
                        x_prompt_recon[i][j, :],
                        sr=sr,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        win_length=win_length,
                    )
                )
            waveform_gt = np.concatenate(waveform_gt, axis=0)
            waveform_recon = np.concatenate(waveform_recon, axis=0)
            write(
                os.path.join(self.output_dir, f"x_prompt_gt_{i}.wav"),
                waveform_gt,
                sr,
            )
            write(
                os.path.join(self.output_dir, f"x_prompt_recon_{i}.wav"),
                waveform_recon,
                sr,
            )

        print(
            f"Saved {n_samples} x-level inputs and reconstructions to {self.output_dir}"
        )

    def future_pred_waveform(self, n_samples=10):
        """
        Save some x-level future predictions for checking.
        """
        x_future_pred = self.x_future_pred

        config = self.config
        if config["dataloader"] == "insnotes_dataloader":
            sr = 16000
            hop_length = 256
            win_length = 1024
            n_fft = 1024
            # remove the added channel dimension
            x_future_pred = x_future_pred.squeeze()

        for i in range(n_samples):
            waveform_pred = []
            for j in range(x_future_pred[i].shape[0]):
                waveform_pred.append(
                    lr.feature.inverse.mel_to_audio(
                        x_future_pred[i][j, :],
                        sr=sr,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        win_length=win_length,
                    )
                )
            waveform_pred = np.concatenate(waveform_pred, axis=0)
            write(
                os.path.join(self.output_dir, f"x_future_pred_{i}.wav"),
                waveform_pred,
                sr,
            )

        print(f"Saved {n_samples} x-level future predictions to {self.output_dir}")
