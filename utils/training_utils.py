import numpy as np
import torch

import random
import copy


class DDPWithMethods(torch.nn.parallel.DistributedDataParallel):
    """
    Wrap the DistributedDataParallel module to allow access to the module's methods
    """

    def __getattr__(self, name):
        # Redirect attribute access to the wrapped module if it exists there
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = (
        False  # disable cudnn benchmark for reproducibility
    )


def cosine_annealing_with_warmup(
    epoch, lr_anneal_epochs, lr_anneal_min_factor, warmup_epochs, warmup_factor
):
    """
    Cosine annealing with warmup learning rate schedule.
    """
    if epoch < warmup_epochs:
        return warmup_factor
    else:
        return lr_anneal_min_factor + 0.5 * (
            1.0 + np.cos(np.pi * (epoch - warmup_epochs) / lr_anneal_epochs)
        ) * (1.0 - lr_anneal_min_factor)


def exponential_decay_with_warmup(
    epoch,
    lr_decay_factor,
    lr_decay_epochs,
    lr_decay_min_factor,
    warmup_epochs,
    warmup_factor,
):
    """
    Exponential decay with warmup learning rate schedule.
    Decays by lr_decay_factor every lr_decay_epochs epochs.
    When lr smaller than lr_min_factor, lr stays at lr_min_factor.
    """
    if epoch < warmup_epochs:
        return warmup_factor
    else:
        return max(
            lr_decay_min_factor,
            lr_decay_factor ** ((epoch - warmup_epochs) / lr_decay_epochs),
        )


def compute_loss_weight(loss_str, t):
    """
    Compute dynamic weights for the losses based on the current training step t, for curriculum learning.
    loss_str looks like this (yaml):
    loss_str: "some_function(t, blabla)"
    """
    try:
        weight = eval(loss_str, {"t": t})
    except Exception as e:
        raise ValueError(f"Error evaluating weight: {e}")

    return weight


def compute_loss_weights(loss_config, t):
    """
    Compute dynamic weights for the losses based on the current training step t, for curriculum learning.
    loss_config looks like this (yaml):
    loss_config:
      weights:
        xxx: 1.0
        yyy: 1.0
        zzz: "some_function(t, blabla)"
      blabla:
    """
    weights = loss_config.get("weights", {})  # Extract the weight definitions
    computed_weights = {}

    for loss_name, weight_expr in weights.items():
        if isinstance(
            weight_expr, (int, float)
        ):  # If it's a fixed number, use it directly
            computed_weights[loss_name] = weight_expr
        elif isinstance(weight_expr, str):  # If it's a function expression, evaluate it
            try:
                computed_weights[loss_name] = eval(weight_expr, {"t": t, **loss_config})
            except Exception as e:
                raise ValueError(f"Error evaluating weight for {loss_name}: {e}")
        else:
            raise ValueError(f"Invalid weight format for {loss_name}: {weight_expr}")

    return computed_weights


class PCGrad:
    def __init__(self, optimizer, reduction="mean"):
        """
        PCGrad resolves gradient conflicts by modifying the gradients before updating model parameters.
        """
        self._optim, self._reduction = optimizer, reduction
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        """
        clear the gradient of the parameters
        """
        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        """
        update the parameters with the gradient
        """
        return self._optim.step()

    def pc_backward(self, objectives):
        """
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives, or a dictionary of objectives
        """
        if isinstance(objectives, dict):
            objectives = list(objectives.values())
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).mean(dim=0)
        elif self._reduction == "sum":
            merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).sum(dim=0)
        else:
            exit("invalid reduction method")

        merged_grad[~shared] = torch.stack([g[~shared] for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        """
        set the modified gradients to the network
        """
        idx = 0
        for group in self._optim.param_groups:
            for p in group["params"]:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        """
        pack the gradient of the parameters of the network for each objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        """
        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx : idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        """
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        """

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group["params"]:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    epochs = 500

    lr = []
    for epoch in range(epochs):
        lr.append(
            exponential_decay_with_warmup(
                epoch,
                lr_decay_factor=0.1,
                lr_decay_epochs=200,
                lr_decay_min_factor=0.01,
                warmup_epochs=10,
                warmup_factor=0.01,
            )
        )

    plt.plot(lr)
    plt.yscale("log")
    plt.savefig("lr.png")
