import sys
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('AGG')
import torch
import numpy as np
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from typing import List
from VQ.VQVAE import VQVAE
from shared import *
import os
from importlib import reload


CURR_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_ROOT = os.path.join(CURR_DIR, "dec_vis_eval_2digit")


class ExpGroup:
    def __init__(
            self,
            exp_name: str,
            sub_exp: str,
            check_point: str = '',
            exp_root: str = os.path.join(CURR_DIR, '../VQ/exp'),
            result_path: str = "temporal.png",
            ):
        os.makedirs(RECORD_ROOT, exist_ok=True)
        self.exp_root = exp_root
        self.exp_name = exp_name
        self.sub_exp = sub_exp
        self.check_point = check_point
        self.result_path = os.path.join(RECORD_ROOT, result_path)
        self.model, self.digit_num, self.emb_dim, self.dict_size = self.init_model()

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
        return model, digit_num, emb_dim, dict_size

    def run_eval(self):
        plot_dec_img(self.model, self.dict_size, self.digit_num, self.emb_dim, self.result_path)



def flat_idx_list(dict_size, digit_num):
    total_decimal_num = list(range(0, pow(dict_size, digit_num)))
    total_digit_num = [decimal_to_base(i, dict_size) for i in total_decimal_num]
    for i in range(0, len(total_digit_num)):
        while len(total_digit_num[i]) < digit_num:
            total_digit_num[i] = f'0{total_digit_num[i]}'
    total_digit_num_list = [[int(i) for i in j] for j in total_digit_num]
    flat_list = [num for sublist in total_digit_num_list for num in sublist]
    return flat_list

def plot_dec_img(loaded_model: VQVAE, dict_size: int, digit_num: int, emb_dim: int, save_path: str):
    flat_idx = torch.tensor(flat_idx_list(dict_size, digit_num), dtype=torch.int).to(DEVICE)
    flat_emb = loaded_model.vq_layer.quantize(flat_idx)
    total_img = pow(dict_size, digit_num)
    total_img_dim = digit_num * emb_dim
    emb = flat_emb.reshape(total_img, total_img_dim).detach()
    imgs = loaded_model.batch_decode_from_z(emb).detach()
    n_row = 1 if digit_num == 1 else dict_size
    imgs2D = imgs.reshape(n_row, dict_size, *imgs.size()[1:])
    print(flat_idx)
    tensor2imgs(imgs2D, save_path)

def tensor2imgs(imgs, save_path):
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
    for ax in axs.flat:
        ax.label_outer()
    plt.suptitle("View from decoder")
    plt.savefig(save_path)
    plt.cla()
    plt.clf()
    plt.close()

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
