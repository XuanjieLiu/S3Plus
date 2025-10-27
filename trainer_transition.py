"""
Pure engineering code for training.
The only algorithmic part is the optimizer.
"""

import os
import gc
import sys
import datetime
import logging
import torch.distributed
import itertools
import yaml
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from matplotlib import pyplot as plt
import seaborn as sns
import wandb

from utils.training_utils import *
from utils.eval_utils import *

from trainer import Trainer


class TrainerTransition(Trainer):
    def __init__(self, config):
        super().__init__(config)

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
            # data_dir=os.path.join(self.data_dir, "train"),
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            data_type=config["data_type"],
            # shuffle=True,  # when distributed, shuffle is omitted
            # distributed=False,
            mode="major",
        )
        logging.info("Train dataloader ready.")

        self.val_loader = dataloader_module.get_dataloader(
            data_dir=os.path.join(self.data_dir),
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            data_type=config["data_type"],
            # shuffle=False,  # when distributed, shuffle is omitted
            test=True,
            # distributed=False,
        )
        # check how many steps in the validation dataloader if using "whole" as the validation strategy
        if config["val_steps"] == "whole":
            self.config["val_steps"] = len(self.val_loader)
        self.val_loader = itertools.cycle(self.val_loader)  # cycle for validation
        logging.info("Validation dataloader ready.")

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
                if "Transition" in method_specs:
                    Model = import_module(
                        "model.simple_rnn_insnotes_transition"
                    ).SymmCSAEwithTransition
                    Loss = import_module(
                        "model.symm_loss_transition"
                    ).SymmLossTransition
                else:
                    Model = import_module("model.simple_rnn_insnotes").SymmCSAEwithPrior
                    Loss = import_module("model.symm_loss").SymmLoss

        model_config = self.config["model_config"]
        optimizer_config = self.config["optimizer_config"]
        loss_config = self.config["loss_config"]

        # model
        self.model = Model(model_config).to(self.device)
        # self.model = DDPWithMethods(
        #     self.model,
        #     device_ids=[self.local_rank],
        #     output_device=self.local_rank,
        #     find_unused_parameters=False,
        # )
        logging.info("Model set up.")
        logging.info(self.model.get_model_size())

        # optimizer
        if optimizer_config["optimizer"] == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config["lr"],
                betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
                weight_decay=optimizer_config["weight_decay"],
            )
        elif optimizer_config["optimizer"] == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_config["lr"],
                momentum=optimizer_config["momentum"],
                weight_decay=optimizer_config["weight_decay"],
            )
        logging.info(f"Optimizer {optimizer_config['optimizer']} set up.")

        # loss function
        self.loss = Loss(loss_config, use_isymm="ISymm" in method_specs)

        # load previous model
        self.start_step = 0
        if "load_checkpoint" in self.config and self.config["load_checkpoint"]:
            cp_path = self.config["load_checkpoint"]
            if os.path.exists(cp_path):
                save_info = torch.load(cp_path)
                self.start_step = save_info["step"]
                self.model.load_state_dict(save_info["model"])
                self.optimizer.load_state_dict(save_info["optimizer"])
                logging.info(
                    f"Checkpoint loaded from {cp_path} at step {self.start_step}."
                )
            else:
                logging.info(
                    f"No checkpoint found at {cp_path}. Will start from scratch."
                )

        # scheduler.
        if optimizer_config["scheduler"] == "cosine_annealing":
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda t: cosine_annealing_with_warmup(
                    t=t,
                    lr_anneal_steps=optimizer_config["lr_anneal_steps"],
                    lr_anneal_min_factor=optimizer_config["lr_anneal_min_factor"],
                    lr_anneal_restart_decay_factor=optimizer_config[
                        "lr_anneal_restart_decay_factor"
                    ],
                    warmup_steps=optimizer_config["warmup_steps"],
                    warmup_factor=optimizer_config["warmup_factor"],
                ),
                last_epoch=self.start_step
                - 1,  # important for resuming training. the keyword is "last_epoch" but it's actually "last_step"
            )
        elif optimizer_config["scheduler"] == "exponential_decay":
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda t: exponential_decay_with_warmup(
                    t=t,
                    lr_decay_factor=optimizer_config["lr_decay_factor"],
                    lr_decay_steps=optimizer_config["lr_decay_steps"],
                    lr_decay_min_factor=optimizer_config["lr_decay_min_factor"],
                    warmup_steps=optimizer_config["warmup_steps"],
                    warmup_factor=optimizer_config["warmup_factor"],
                ),
                last_epoch=self.start_step - 1,  # important for resuming training
            )
        logging.info(f"Scheduler {optimizer_config['scheduler']} set up.")

        # PCGrad
        if "PCGrad" in optimizer_config and optimizer_config["PCGrad"]:
            self.pcgrad = PCGrad(self.optimizer, reduction="mean")
            logging.info("PCGrad set up.")
        else:
            self.pcgrad = None

    def train(self):
        """
        training and validation loop.
        """
        config = self.config
        n_steps = config["steps"]
        step = 0

        self.model, self.optimizer, self.train_loader, self.val_loader = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.val_loader
            )
        )

        self.model.train()
        running_losses_train = {}
        train_loader = self.train_loader


        while step < n_steps:
            # training loop

            batch_data, c_labels, s_labels = next(train_loader)
            # Move data to device
            batch_data = batch_data.to(device=self.device)
            # forward
            with self.accelerator.autocast():
                losses = self.loss.compute_loss(step, self.model, batch_data)
            # backward
            self.optimizer.zero_grad(set_to_none=True)
            # if self.pcgrad is not None:
            #     pc_grad_losses = [x[1] for x in losses.items() if x[0] != "total_loss"]
            #     pc_grad_losses = [self.scaler.scale(x) for x in pc_grad_losses]
            #     self.pcgrad.pc_backward(
            #         pc_grad_losses,
            #     )
            # else:
            self.accelerator.backward(losses["total_loss"])
            # grad clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

            # optimizer step
            self.optimizer.step()

            # accumulate running loss
            for k, v in losses.items():
                if k not in running_losses_train:
                    running_losses_train[k] = 0
                running_losses_train[k] += v.item()

            # write to log
            if step % config["log_every_n_steps"] == 0:
                for k, v in running_losses_train.items():
                    running_losses_train[k] /= config["log_every_n_steps"]
                logging.info(
                    f"TRAIN - Step {step}, Loss: {running_losses_train['total_loss']:.4f}"
                )
                # write summary for this log cycle
                self._write_summary(
                    step + self.start_step, running_losses_train, "train"
                )
                running_losses_train = {}

            # validation loop
            if (
                step % config["val_every_n_steps"] == 1
            ):  # make sure to validate after 1 step
                self.model.eval()
                running_losses_val = {}
                val_loader = self.val_loader
                collect_c_labels = []
                collect_c_preds = []
                for i in range(config["val_steps"]):
                    batch_data, c_labels, s_labels = next(val_loader)
                    # Move data to device
                    batch_data = batch_data.to(device=self.device)
                    # forward
                    with torch.no_grad():
                        losses = self.loss.compute_loss(
                            step, self.model, batch_data, is_train=False
                        )
                        # accumulate running loss
                        for k, v in losses.items():
                            if k not in running_losses_val:
                                running_losses_val[k] = 0
                            running_losses_val[k] += v.item()
                        zc_vq, indices, commit_loss, zs = self.model.encode(
                            batch_data, quantize=True
                        )
                        collect_c_labels.append(c_labels.cpu().numpy())
                        collect_c_preds.append(indices.cpu().numpy())

                # write to log
                for k, v in running_losses_val.items():
                    running_losses_val[k] /= config["val_steps"]
                logging.info(
                    f"VALIDATION - Step {step}, Loss: {running_losses_val['total_loss']:.4f}"
                )
                # write summary for this validation cycle
                # plot recon
                x_hat, zc, zc_vq, indices, commit_loss, zs = self.model(batch_data)
                x = batch_data[0][0].detach().cpu().numpy()
                x_hat = x_hat[0][0].detach().cpu().numpy()
                c = int(c_labels[0][0].detach().cpu().numpy())
                s = int(s_labels[0][0].detach().cpu().numpy())

                x = x.reshape((x.shape[1], x.shape[2])).T
                x_hat = x_hat.reshape((x_hat.shape[1], x_hat.shape[2])).T

                # x = x.reshape((x.shape[2], x.shape[0] * x.shape[3])).T
                # x_hat = x_hat.reshape(
                #     (x_hat.shape[2], x_hat.shape[0] * x_hat.shape[3])
                # ).T
                x_plot = plot_spectrogram(x, c, s)
                x_hat_plot = plot_spectrogram(x_hat, c, s)

                # plot AE content confusion mtx
                collect_c_labels = np.concatenate(collect_c_labels, axis=0).flatten()
                collect_c_preds = np.concatenate(collect_c_preds, axis=0).flatten()
                confusion_mtx, perm = get_confusion_mtx(
                    config["model_config"]["n_atoms"],
                    12,
                    collect_c_preds,
                    collect_c_labels,
                )
                confusion_plot = plot_confusion_mtx(
                    confusion_mtx,
                    title="AE content confusion matrix",
                )

                # plot transition mtx
                with torch.no_grad():
                    codebook = self.model.vq.codebook  # (n_atoms, d_zc)
                    energies = self.model.compute_energies(
                        codebook, codebook
                    )  # (n_atoms, n_atoms)
                transition_mtx = (
                    F.softmax(energies / 0.3, dim=-1).detach().cpu().numpy()
                )
                transition_plot = plot_confusion_mtx(
                    transition_mtx,
                    title="Transition probability matrix",
                )

                self._write_summary(
                    step + self.start_step,
                    running_losses_val,
                    "val",
                    fig=[x_plot, x_hat_plot, confusion_plot, transition_plot],
                )

                # checkpoint
                if step % config["save_every_n_steps"] == 1:
                    self._save_checkpoint(step, running_losses_val["total_loss"])

                gc.collect()
                self.model.train()

            # scheduler step
            step += 1
            self.scheduler.step()

    def _write_summary(self, i_step, losses, partition="train", fig=None, plot=None):
        if self.config["debug"]:
            return
        log_dict = {}
        for k, v in losses.items():
            log_dict[f"{partition}/{k}"] = v
        if partition == "val":
            log_dict["lr"] = self.optimizer.param_groups[0]["lr"]
        wandb.log(log_dict, step=i_step)
        if fig is not None:
            if isinstance(fig, list):
                for i, f in enumerate(fig):
                    if f is not None:
                        wandb.log({f"{partition}/fig_{i}": wandb.Image(f)}, step=i_step)
            elif isinstance(fig, np.ndarray):
                wandb.log({f"{partition}/fig": wandb.Image(fig)}, step=i_step)
        if plot is not None:
            if isinstance(plot, list):
                for i, p in enumerate(plot):
                    if p is not None:
                        wandb.log(
                            {f"{partition}/plot_{i}": wandb.Plotly(p)}, step=i_step
                        )
            elif isinstance(plot, plt.Figure):
                wandb.log({f"{partition}/plot": wandb.Image(plot)}, step=i_step)

    def _save_checkpoint(self, step, val_loss):
        # keep the best checkpoints
        self.performance_history = self.performance_history or {}
        if len(self.performance_history) > self.config["save_top_k"]:
            # remove the worst checkpoint
            worst_step = max(self.performance_history, key=self.performance_history.get)
            worst_path = os.path.join(self.log_dir, f"cp_step{worst_step}.pt")
            os.remove(worst_path)
            del self.performance_history[worst_step]

        # save the current checkpoint
        save_info = {
            "step": step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        save_name = f"cp_step{step}.pt"
        save_path = os.path.join(self.log_dir, save_name)
        torch.save(save_info, save_path)
        self.performance_history[step] = val_loss
        logging.info(f"Checkpoint saved at {save_path}")
