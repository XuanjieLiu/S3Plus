import sys
import os
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader

from dataloader import SingleImgDataset

matplotlib.use('AGG')
import torch
import numpy as np
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from typing import List
from VQ.VQVAE import VQVAE
from shared import *
import os
from importlib import reload
from VQ.num_eval import load_enc_eval_data
from functools import reduce

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_ROOT = os.path.join(CURR_DIR, "dec_vis_eval_2digit")


class ExpGroup:
    def __init__(
            self,
            exp_name: str,
            sub_exp: str,
            check_point: str = '',
            exp_root: str = os.path.join(CURR_DIR, '../VQ/exp'),
            result_path: str = "result.png",
            ):
        os.makedirs(RECORD_ROOT, exist_ok=True)
        self.exp_root = exp_root
        self.exp_name = exp_name
        self.sub_exp = sub_exp
        self.check_point = check_point
        result_name = f'{check_point}.{result_path}'
        self.result_path = os.path.join(RECORD_ROOT, result_name)
        self.model, self.digit_num, self.emb_dim, self.dict_size, self.eval_loader_1, self.multi_num_embeddings = self.init_model()

    def init_model(self):
        tar_exp_path = os.path.join(self.exp_root, self.exp_name)
        os.chdir(tar_exp_path)
        sys.path.append(tar_exp_path)
        print(f'Exp path: {tar_exp_path}')
        t_config = __import__('train_config')
        reload(t_config)
        sys.path.pop()
        cp_path = os.path.join(tar_exp_path, self.sub_exp, self.check_point)
        model = VQVAE(t_config.CONFIG).to(DEVICE)
        model.load_state_dict(model.load_tensor(cp_path))
        model.eval()
        digit_num = t_config.CONFIG['latent_embedding_1']
        emb_dim = t_config.CONFIG['embedding_dim']
        dict_size = t_config.CONFIG['embeddings_num']
        multi_num_embeddings = t_config.CONFIG['multi_num_embeddings']
        eval_set_1 = SingleImgDataset(t_config.CONFIG['single_img_eval_set_path'])
        eval_loader_1 = DataLoader(eval_set_1, batch_size=32)
        return model, digit_num, emb_dim, dict_size, eval_loader_1, multi_num_embeddings

    def run_eval(self):
        num_z, num_labels = load_enc_eval_data(
            self.eval_loader_1,
            lambda x: self.model.find_indices(
                self.model.batch_encode_to_z(x)[0],
                False
            )
        )
        enc_flat_z = [int(t.item()) for t in num_z]
        plot_dec_img(self.model, self.dict_size, self.digit_num, self.result_path, enc_flat_z, num_labels, self.multi_num_embeddings)


def idx_list(dict_size_list: List[int]):
    total_decimal_num = list(range(0, reduce(lambda x, y: x * y, dict_size_list)))
    total_digit_num = [decimal_to_base(i, dict_size_list[1]) for i in total_decimal_num]
    for i in range(0, len(total_digit_num)):
        while len(total_digit_num[i]) < len(dict_size_list):
            total_digit_num[i] = f'0{total_digit_num[i]}'
    total_digit_num_list = [[int(i) for i in j] for j in total_digit_num]
    return total_digit_num_list


def plot_dec_img(
        loaded_model: VQVAE,
        dict_size: int,
        digit_num: int,
        save_path: str,
        enc_flat_z,
        enc_labels,
        dict_sizes: List[int] = None
):
    if dict_sizes is None:
        dict_size_list = [dict_size for i in range(digit_num)]
    else:
        dict_size_list = dict_sizes
    idx = torch.tensor(idx_list(dict_size_list), dtype=torch.int).to(DEVICE)
    emb = loaded_model.vq_layer.quantize(idx).detach()
    imgs = loaded_model.batch_decode_from_z(emb).detach()
    n_row = dict_size_list[0]
    n_col = dict_size_list[1]
    imgs2D = imgs.reshape(n_row, n_col, *imgs.size()[1:])
    tensor2imgs(imgs2D, save_path, enc_flat_z, enc_labels)


def tensor2imgs(imgs, save_path, enc_flat_z, enc_labels):
    n_row = len(imgs)
    n_col = len(imgs[0])
    fig, axs = plt.subplots(n_row, n_col)
    for i in range(0, n_row):
        for j in range(0, n_col):
            axs[i, j].imshow(arrange_tensor(imgs[i][j]))
            axs[i, j].set(ylabel=f"{i}")
            axs[i, j].set(xlabel=f"{j}")
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            label_enc(i, j, axs[i, j], enc_flat_z, enc_labels, n_col)
    for ax in axs.flat:
        ax.label_outer()
    plt.suptitle("View from decoder")
    plt.savefig(save_path)
    plt.cla()
    plt.clf()
    plt.close()


def label_enc(i, j, axs, enc_flat_z: List[int], enc_labels, dict_size):
    num = i*dict_size + j
    if num not in enc_flat_z:
        return
    idx = enc_flat_z.index(num)
    label = enc_labels[idx]
    axs.set_title(f'{label}', y=1.0, pad=-14, fontweight='bold', color=(0.8, 0.2, 0.2, 0.8))


def decimal_to_base(n, base):
    if base < 2 or base > 36:
        raise ValueError("Base must be between 2 and 36")
    if n == 0:
        return "0"
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    result = ""
    while n > 0:
        digit = n % base
        result = digits[digit] + result
        n //= base
    return result


def arrange_tensor(input: torch.Tensor, need_permute=True):
    tensor = input.permute(1, 2, 0) if need_permute else input
    return np.array(tensor.cpu().detach(), dtype=float)


if __name__ == '__main__':
    EXP_ROOT_PATH = os.path.join(CURR_DIR, '../VQ/exp')
    print(EXP_ROOT_PATH)
