import os
import random
from collections import defaultdict
from typing import List, Tuple, Dict

import torch
from torch.utils.data import DataLoader
from VQVAE import VQVAE
from shared import *
from eval_common import draw_scatter_point_line
import matplotlib.markers
import matplotlib.pyplot as plt
from scipy import stats
from common_func import parse_label, solve_label_emb_one2one_matching

matplotlib.use('AGG')
MODEL_PATH = 'curr_model.pt'
EVAL_ROOT = 'eval_z'

MIN_KS_NUM = 1


class EncInfo:
    def __init__(self, label, emb_idx, emb_value=None, enc_style=None):
        self.label = label
        self.emb_idx = emb_idx
        self.emb_value = emb_value
        self.enc_style = enc_style


class PlusInfo:
    def __init__(self, label_a=None, label_b=None, emb_idx=None, cycle_emb_idx=None,
                 emb_value=None, cycle_emb_value=None, style=None, cycle_style=None, label_c=None):
        """
        Represents the result of a plus operation on two embeddings.
        :param label_a: the label of a.
        :param label_b: the label of b.
        :param emb_idx: the emb index of plus operation of a and b.
        :param cycle_emb_idx: the encoded emb index of the reconstruction of emb_value.
        :param emb_value: the value of the sum emb of plus operation.
        :param cycle_emb_value: the value of the reconstruction of emb_value.
        :param style: the style of the plus operation, if applicable.
        :param cycle_style: the style of the reconstruction, if applicable.
        :param label_c: the label of the plus operation result, which is a + b
        """
        self.emb_idx = emb_idx
        self.label_c = label_c if label_c is not None else (label_a + label_b)
        self.cycle_emb_idx = cycle_emb_idx
        self.emb_value = emb_value
        self.cycle_emb_value = cycle_emb_value
        self.style = style
        self.cycle_style = cycle_style
        self.label_a = label_a
        self.label_b = label_b


def plot_plusZ_against_label(
        all_enc_z,
        all_plus_z,
        eval_path,
        n_rows: int = 1,
        n_cols: int = -1,
        is_scatter_lines=False,
        is_gird=False,
        sub_title: str = '',
        y_label: str = '',
        sub_fig_size=5
):
    dim_z = all_enc_z[0].emb_value.size(0)
    n_row = n_rows
    n_col = dim_z if n_cols == -1 else n_cols
    fig, axs = plt.subplots(n_row, n_col, sharey='all', sharex='all',
                            figsize=(n_col * sub_fig_size, n_row * sub_fig_size), squeeze=False)
    enc_x = [ob.label for ob in all_enc_z]
    plus_x = [ob.label_c for ob in all_plus_z]
    for i in range(0, dim_z):
        enc_y = [ob.emb_value.cpu()[i].item() for ob in all_enc_z]
        plus_y = [ob.emb_value.cpu()[i].item() for ob in all_plus_z]
        axs.flat[i].scatter(enc_x, enc_y, edgecolors='blue', label='Encoder output', facecolors='none', s=60)
        axs.flat[i].scatter(plus_x, plus_y, edgecolors='red', label='Plus output', facecolors='none', s=20)
        axs.flat[i].set(xlabel='Num of Points on the card', xticks=range(0, max(enc_x) + 1))
        axs.flat[i].set(ylabel=y_label)
        axs.flat[i].set_title(f"{sub_title} ({i + 1})")
        axs.flat[i].grid(is_gird)
        if is_scatter_lines:
            draw_scatter_point_line(axs.flat[i], [*enc_x, *plus_x], [*enc_y, *plus_y])
    for ax in axs.flat:
        ax.label_outer()
    plt.legend()
    plt.savefig(f'{eval_path}.png')
    plt.cla()
    plt.clf()
    plt.close()


class LabelKs:
    def __init__(self, label: int):
        self.label = label
        self.enc_z_list = []
        self.plus_z_list = []

    def calc_min_len(self):
        return min(len(self.enc_z_list), len(self.plus_z_list))

    def calc_ks(self):
        min_len = self.calc_min_len()
        sample1 = random.sample(self.enc_z_list, min_len)
        sample2 = random.sample(self.plus_z_list, min_len)
        ks, p_value = stats.ks_2samp(sample1, sample2)
        return ks, p_value

    def calc_accu(self):
        total = len(self.plus_z_list)
        assert total != 0, f"plus_z_list of card {self.label} is empty."
        assert len(self.enc_z_list) != 0, f"enc_z_list of card {self.label} is empty."
        mode_enc = stats.mode(self.enc_z_list, keepdims=False)[0]
        correct = len(list(filter(lambda x: x == mode_enc, self.plus_z_list)))
        accu = correct / total
        return accu


def calc_ks_enc_plus_z(enc_z: List[EncInfo], plus_z: List[PlusInfo]):
    min_label = min(enc_z, key=lambda x: x.label).label
    max_label = max(enc_z, key=lambda x: x.label).label
    label_dict = {}
    for i in range(min_label, max_label + 1):
        label_dict[i] = LabelKs(i)
    for item in enc_z:
        label_dict[item.label].enc_z_list.append(item.emb_idx.cpu().item())
    for item in plus_z:
        label_dict[item.label_c].plus_z_list.append(item.emb_idx.cpu().item())
    label_ks_list = list(label_dict.values())
    ks_all = 0
    accu_all = 0
    item_num = 0
    for item in label_ks_list:
        min_len = item.calc_min_len()
        if min_len >= MIN_KS_NUM:
            ks_all += item.calc_ks()[0]
            accu_all += item.calc_accu()
            item_num += 1
    ks_mean = ks_all / item_num
    accu_mean = accu_all / item_num
    return round(ks_mean, 4), round(accu_mean, 4)



def mode_emb_label_dict(enc_z: List[EncInfo]) -> Dict[int, int]:
    """
    根据 enc_z 中每个 embedding 的 label 众数，返回 emb → label 的映射。
    """
    emb_to_labels = defaultdict(list)
    for item in enc_z:
        emb_to_labels[int(item.emb_idx.item())].append(item.label)
    # 取众数
    return {
        emb: int(stats.mode(labels, keepdims=False)[0])
        for emb, labels in emb_to_labels.items()
    }



def one2one_emb_label_dict(enc_z: List[EncInfo]) -> Dict[int, int]:
    """
    调用 Hungarian 算法，返回一对一匹配的 emb → label 映射。
    """
    emb_idx_list = [int(item.emb_idx.item()) for item in enc_z]
    label_list = [item.label for item in enc_z]
    emb_label_pairs, _ = solve_label_emb_one2one_matching(emb_idx_list, label_list)
    # emb_label_pairs 是 (label, emb) 对
    return {emb: int(label) for label, emb in emb_label_pairs}


def plus_accuracy(
        emb_label_dict: Dict[int, int],
        plus_z: List[PlusInfo]
) -> Tuple[float, float]:
    """
    统一的准确率计算逻辑：对 plus_c_z 与 plus_c_z_cycle 两种情况分别计数。
    """
    n_total = len(plus_z)
    if n_total == 0:
        raise ValueError("plus_z is empty.")
    n_correct = 0
    n_correct_cycle = 0

    for item in plus_z:
        true_label = int(item.label_c)
        z1 = int(item.emb_idx.item())
        z2 = int(item.cycle_emb_idx.item())
        if emb_label_dict.get(z1) == true_label:
            n_correct += 1
        if emb_label_dict.get(z2) == true_label:
            n_correct_cycle += 1

    return n_correct / n_total, n_correct_cycle / n_total


def calc_multi_emb_plus_accu(
        enc_z: List[EncInfo],
        plus_z: List[PlusInfo]
) -> Tuple[float, float]:
    """
    多 Embedding → 多 Label 情形：embedding 通过众数关联 label。
    """
    emb_label_dict = mode_emb_label_dict(enc_z)
    return plus_accuracy(emb_label_dict, plus_z)


def calc_one2one_plus_accu(
        enc_z: List[EncInfo],
        plus_z: List[PlusInfo]
) -> Tuple[float, float]:
    """
    一对一情形：使用 Hungarian 算法匹配 embedding 与 label。
    """
    emb_label_dict = one2one_emb_label_dict(enc_z)
    return plus_accuracy(emb_label_dict, plus_z)


def calc_plus_z_self_cycle_consistency(plus_z: List[PlusInfo]):
    """
    通过比较 plus_c_z 和 plus_c_z_cycle 是否相等来，
    计算 plus_z 中的 plus_c_z 与 plus_c_z_cycle 的一致性。
    """
    assert len(plus_z) != 0, "plus_z is empty."
    n_total = len(plus_z)
    n_consistent = 0
    for item in plus_z:
        if int(item.emb_idx.item()) == int(item.cycle_emb_idx.item()):
            n_consistent += 1
    return n_consistent / n_total


def calc_plus_z_mode_emb_label_cycle_consistency(enc_z: List[EncInfo], plus_z: List[PlusInfo]):
    """
    计算 plus_z 中的 plus_c_z 与 plus_c_z_cycle 的一致性。
    通过 enc_z 中的 embedding 众数来确定每个 embedding 对应的 label。
    根据判断 plus_c_z 和 plus_c_z_cycle 是否对应同一个 label 计算 consistency。
    :param enc_z: List of EncZ objects, each containing a label and an embedding index.
    :param plus_z: List of PlusZ objects, each containing a label and embedding indices for the plus operation.
    :return: Tuple of three floats:
        - consistency: float, the proportion of plus_c_z and plus_c_z_cycle that match the same label.
        - recognized_c_z: float, the proportion of plus_c_z that are recognized by enc_z.
        - recognized_c_z_cycle: float, the proportion of plus_c_z_cycle that are recognized by enc_z.
    """
    assert len(plus_z) != 0, "plus_z is empty."
    assert len(enc_z) != 0, "enc_z is empty."
    emb_label_dict = mode_emb_label_dict(enc_z)
    n_consistent = 0
    n_consistent_total = 0
    n_total = len(plus_z)
    n_recognized_c_z = 0
    n_recognized_c_z_cycle = 0
    for item in plus_z:
        z1 = int(item.emb_idx.item())
        z2 = int(item.cycle_emb_idx.item())
        label1 = emb_label_dict.get(z1)
        label2 = emb_label_dict.get(z2)
        if label1 is not None:
            n_recognized_c_z += 1
        if label2 is not None:
            n_recognized_c_z_cycle += 1
        if label1 is not None and label2 is not None:
            n_consistent_total += 1
            if label1 == label2:
                n_consistent += 1
    consistent_rate = n_consistent / n_consistent_total if n_consistent_total > 0 else 0
    return consistent_rate, n_recognized_c_z / n_total, n_recognized_c_z_cycle / n_total


class VQvaePlusEval:
    def __init__(self, config, model_path=None, loaded_model: VQVAE = None):
        self.config = config
        self.zc_dim = config['latent_embedding_1'] * config['embedding_dim']
        if loaded_model is not None:
            self.model = loaded_model
        elif model_path is not None:
            self.model = VQVAE(config).to(DEVICE)
            self.reload_model(model_path)
            print(f"Model is loaded from {model_path}")
        else:
            self.model = VQVAE(config).to(DEVICE)
            print("No model is loaded")
        self.model.eval()
        self.isVQStyle = config['isVQStyle']

    def reload_model(self, model_path):
        self.model.load_state_dict(self.model.load_tensor(model_path))

    def load_plusZ_eval_data(self, data_loader: DataLoader):
        all_enc_z = []
        all_plus_z = []
        for batch_ndx, sample in enumerate(data_loader):
            enc_z_list = []
            plus_z_list = []
            data, labels = sample
            data = [x.to(DEVICE) for x in data]
            za = self.model.batch_encode_to_z(data[0])[0]
            zb = self.model.batch_encode_to_z(data[1])[0]
            zc = self.model.batch_encode_to_z(data[2])[0]
            za_s = za[..., self.zc_dim:]
            zb_s = zb[..., self.zc_dim:]
            zc_s = zc[..., self.zc_dim:]
            if self.isVQStyle:
                za_s = self.model.vq_layer(za_s)[0]
                zb_s = self.model.vq_layer(zb_s)[0]
                zc_s = self.model.vq_layer(zc_s)[0]
            za = za[..., 0:self.zc_dim]
            zb = zb[..., 0:self.zc_dim]
            zc = zc[..., 0:self.zc_dim]
            label_a = [parse_label(x) for x in labels[0]]
            label_b = [parse_label(x) for x in labels[1]]
            label_c = [parse_label(x) for x in labels[2]]
            plus_c = self.model.plus(za, zb)[0]
            plus_c_cycle = self.model.batch_encode_to_z(
                self.model.batch_decode_from_z(torch.cat((plus_c, za_s), -1))
            )[0]
            plus_c_cycle_style = plus_c_cycle[..., self.zc_dim:]
            if self.isVQStyle:
                plus_c_cycle_style = self.model.vq_layer(plus_c_cycle_style)[0]
            plus_c_cycle = plus_c_cycle[..., 0:self.zc_dim]

            idx_z_a = self.model.find_indices(za, False)
            idx_z_b = self.model.find_indices(zb, False)
            idx_z_c = self.model.find_indices(zc, False)
            idx_plus_c = self.model.find_indices(plus_c, False)
            idx_plus_c_cycle = self.model.find_indices(plus_c_cycle, False)

            for i in range(0, za.size(0)):
                enc_z_list.append(EncInfo(label_a[i], idx_z_a[i], za[i], za_s[i]))
                enc_z_list.append(EncInfo(label_b[i], idx_z_b[i], zb[i], zb_s[i]))
                enc_z_list.append(EncInfo(label_c[i], idx_z_c[i], zc[i], zc_s[i]))
                plus_z_list.append(PlusInfo(
                    label_a[i], label_b[i], idx_plus_c[i], idx_plus_c_cycle[i],
                    zc[i], plus_c_cycle[i], za_s[i], plus_c_cycle_style[i]
                ))
            all_enc_z.extend(enc_z_list)
            all_plus_z.extend(plus_z_list)
        return all_enc_z, all_plus_z


def get_all_plus_pairs(plus_z: List[PlusInfo]) -> Tuple[List[int], List[int], List[int]]:
    """
    从 plus_z 中提取所有不重复的的 a, b, c 对。
    :param plus_z: List of PlusInfo objects.
    :return: Tuple of three lists: a, b, c.
    """
    a = []
    b = []
    c = []
    # 使用 set 来避免重复的 a-b 对
    a_b_strings = set()
    for item in plus_z:
        a_b_str = f"{item.label_a}-{item.label_b}"
        if a_b_str not in a_b_strings:
            a_b_strings.add(a_b_str)
            a.append(item.label_a)
            b.append(item.label_b)
            c.append(item.label_c)
    return a, b, c


def interpolate_plus_eval(
    loaded_model: VQVAE,
    enc_z: List[EncInfo],
    plus_z: List[PlusInfo],
    n_trials: int = 10,
):
    """
    执行 enc_z 的插值加法评估。
    """
    half = 0.5
    a_list, b_list, c_list = get_all_plus_pairs(plus_z)
    smallest = min(min(a_list), min(b_list))
    largest = max(max(a_list), max(b_list))
    accu_list, accu_cycle_list = [], []

    for i in range(n_trials):
        pair_strings = set()
        a_itp_list, b_itp_list, style_enc_list, label_c = [], [], [], []
        def is_valid_pair(a, b):
            return smallest < a < largest and smallest < b < largest and f"{a}-{b}" not in pair_strings

        for a, b, c in zip(a_list, b_list, c_list):
            a_m = a - half
            b_p = b + half
            if is_valid_pair(a_m, b_p):
                pair_strings.add(f"{a_m}-{b_p}")
                a_itp, b_itp, style_enc = _get_interpolate_plus_pairs(
                    a-1, a, b, b+1, enc_z
                )
                a_itp_list.append(a_itp)
                b_itp_list.append(b_itp)
                style_enc_list.append(style_enc)
                label_c.append(c)

            a_p = a + half
            b_m = b - half
            if is_valid_pair(a_p, b_m):
                pair_strings.add(f"{a_p}-{b_m}")
                a_itp, b_itp, style_enc = _get_interpolate_plus_pairs(
                    a+1, a, b, b-1, enc_z
                )
                a_itp_list.append(a_itp)
                b_itp_list.append(b_itp)
                style_enc_list.append(style_enc)
                label_c.append(c)

        a_itp_tensor = torch.stack(a_itp_list, dim=0).to(DEVICE)
        b_itp_tensor = torch.stack(b_itp_list, dim=0).to(DEVICE)
        style_enc_tensor = torch.stack(style_enc_list, dim=0).to(DEVICE)
        plus_emb = loaded_model.plus(a_itp_tensor, b_itp_tensor)[0]
        plus_emb_idx = loaded_model.find_indices(plus_emb, False)
        plus_emb_cycle = loaded_model.batch_encode_to_z(
            loaded_model.batch_decode_from_z(torch.cat((plus_emb, style_enc_tensor), -1))
        )[0]
        plus_emb_cycle_idx = loaded_model.find_indices(plus_emb_cycle, True)
        plus_itp_list = []
        for j in range(len(plus_emb_idx)):
            plus_itp_list.append(PlusInfo(
                emb_idx=plus_emb_idx[j],
                cycle_emb_idx=plus_emb_cycle_idx[j],
                label_c=label_c[j],
            ))
        emb_label_dict = mode_emb_label_dict(enc_z)
        accu, accu_cycle = plus_accuracy(emb_label_dict, plus_itp_list)
        accu_list.append(accu)
        accu_cycle_list.append(accu_cycle)
    accu_mean = sum(accu_list) / len(accu_list)
    accu_cycle_mean = sum(accu_cycle_list) / len(accu_cycle_list)
    print(f"Interpolated Plus Accu: {accu_mean:.4f}, Cycle Accu: {accu_cycle_mean:.4f}")
    return accu_mean, accu_cycle_mean


def _sample_EncInfo_by_label(
        enc_z: List[EncInfo],
        label: int,
) -> EncInfo:
    """
    从 enc_z 中随机抽取一个 EncInfo 对象，基于给定的 label。
    :param enc_z: List of EncInfo objects.
    :param label: The label to filter EncInfo objects.
    :return: a sampled EncInfo object.
    """
    filtered_enc_z = [item for item in enc_z if int(item.label) == label]
    return random.sample(filtered_enc_z, 1)[0]


def _get_interpolate_plus_pairs(a1: int, a2: int, b1: int, b2: int, enc_z: List[EncInfo]):
    """
    从 enc_z 中获取 a1, a2, b1, b2 的插值对。
    计算 a1 和 a2 的中间点，以及 b1 和 b2 的中间点。
    并随机选择一个样式。
    """
    a1_enc = _sample_EncInfo_by_label(enc_z, a1)
    a2_enc = _sample_EncInfo_by_label(enc_z, a2)
    b1_enc = _sample_EncInfo_by_label(enc_z, b1)
    b2_enc = _sample_EncInfo_by_label(enc_z, b2)
    a_itp = (a1_enc.emb_value + a2_enc.emb_value) * 0.5
    b_itp = (b1_enc.emb_value + b2_enc.emb_value) * 0.5
    style_enc = random.choice([a1_enc.enc_style, a2_enc.enc_style, b1_enc.enc_style, b2_enc.enc_style])
    return a_itp, b_itp, style_enc




