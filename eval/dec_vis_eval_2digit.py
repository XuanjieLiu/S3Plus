import sys
import os

import torch

sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from typing import List
from loss_counter import read_record
from VQ.VQVAE import VQVAE
from shared import *


class ExpGroup:
    def __init__(
            self,
            exp_name: str,
            exp_alias: str,
            sub_exp: str,
            record_name: str,
            exp_root: str = '../VQ/exp',
            check_point: str = ''
            ):
        self.exp_name = exp_name
        self.sub_exps = sub_exp
        self.check_point = check_point
        self.exp_alias = exp_alias
        self.record_name = record_name
        self.file_name = os.path.join(exp_name, sub_exp, check_point)
        self.check_point_path = os.path.join(exp_root, self.file_name)


def plot_dec_img(loaded_model: VQVAE, dict_size: int, digit_num: int, emb_dim: int, save_path: str):
    flat_idx = torch.tensor(flat_idx_list(dict_size, digit_num)).to(DEVICE)

    print(flat_idx)
    return


def flat_idx_list(dict_size, digit_num):
    total_decimal_num = list(range(0, pow(dict_size, digit_num)))
    total_digit_num = [decimal_to_base(i, dict_size) for i in total_decimal_num]
    for i in range(0, len(total_digit_num)):
        while len(total_digit_num[i]) < digit_num:
            total_digit_num[i] = f'0{total_digit_num[i]}'
    total_digit_num_list = [[int(i) for i in j] for j in total_digit_num]
    flat_list = [num for sublist in total_digit_num_list for num in sublist]
    print(flat_list)
    return

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


if __name__ == '__main__':
    print(flat_idx_list(5,2))
