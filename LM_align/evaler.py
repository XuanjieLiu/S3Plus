import torch
import numpy as np
import sys
import os
import json
from typing import Dict, Tuple, Literal, Any
from importlib import reload
from tqdm import tqdm
from torch.utils.data import DataLoader

from collections import Counter
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../../'))
from synthData.SynthImgsDataset import init_test_online_synth_dataloaders, init_test_online_synth_dataloaders_paired
from train_align import AlignTrain
from synthData.SynthImgsDataset import PlanRenderDataset, PreGeneratedDataset, onlineGenDataset, OBJ_LIST_2, OBJ_LIST


LabelKey = Literal["a", "b", "c", "all"]
def _to_list(x: Any):
    """把 batch 中的张量/列表安全地转换成 Python list[int]。"""
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, list) or isinstance(x, tuple):
        return list(x)
    # 单个标量
    return [int(x)]

def count_numeric_labels_per_epoch(
    dataloader,
    key: LabelKey = "all"
) -> Tuple[Dict[int, int], Dict[int, float]]:
    """
    统计某个 dataloader 在一个 epoch 中的数字label分布。

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        承载样本的 DataLoader。其 batch 应包含键 'label'，
        而 'label' 内应含有整数型的 'a'、'b'、'c' 字段。
    key : {'a','b','c','all'}
        - 'a' / 'b' / 'c'：只统计对应字段
        - 'all'：把 a、b、c 三个字段拼在一起统一统计（默认）

    Returns
    -------
    counts : Dict[int, int]
        每个数字label出现的次数（整型字典）。
    ratios : Dict[int, float]
        每个数字label出现的比例（counts 归一化得到）。
    """
    counter = Counter()

    for batch in dataloader:
        labels = batch["label"]  # 可能是“字典的批”，也可能是“由若干字典组成的列表”
        if isinstance(labels, dict):
            # PyTorch 默认 collate：'a','b','c' 可能已经是 LongTensor[batch]
            if key == "all":
                for k in ("a", "b", "c"):
                    vals = _to_list(labels[k])
                    for v in vals:
                        counter[int(v)] += 1
            else:
                vals = _to_list(labels[key])
                for v in vals:
                    counter[int(v)] += 1
        else:
            # 一些自定义 collate 会给出 list[dict]
            # labels 是 list of dict: [{"a":int,"b":int,"c":int,...}, ...]
            for ld in labels:
                if key == "all":
                    for k in ("a", "b", "c"):
                        counter[int(ld[k])] += 1
                else:
                    counter[int(ld[key])] += 1

    total = sum(counter.values()) if counter else 0
    counts = dict(sorted(counter.items(), key=lambda kv: kv[0]))
    ratios = {k: (v / total if total > 0 else 0.0) for k, v in counts.items()}
    return counts, ratios


def upsert_json(json_path: str, big_dict: dict):
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data.update(big_dict)  # 只覆盖/新增 big_dict 中的字段（如 label_acc / obj_acc）
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def select_exps_by_train_acc(json_path: str, acc_threshold: float) -> Dict[str, int]:
    """
    读取指定 json 文件，返回所有 Train_record_accuracy > acc_threshold 的 exp 及其 ckpt。
    返回格式：{ "exp_2": 26, ... }
    """
    def parse_list_of_kv_str(items):
        # 将 ["exp_1: 0.221", "exp_2: 0.717", ...] 解析为 {"exp_1": 0.221, ...}
        out = {}
        for s in items:
            key, val = s.split(":", 1)
            key = key.strip()
            val = val.strip()
            # 尝试转成 int，再不行转 float；两者都不行就原样保留
            try:
                num = int(val)
            except ValueError:
                num = float(val)
            out[key] = num
        return out

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ckpt_map = parse_list_of_kv_str(data["ckpts"])
    train_acc_map = parse_list_of_kv_str(data["Train_record_accuracy"])

    # 只选择同时在 ckpt_map 和 train_acc_map 中的 exp，且 train_acc > 阈值
    result = {
        exp: int(ckpt_map[exp])
        for exp, acc in train_acc_map.items()
        if exp in ckpt_map and acc > acc_threshold
    }
    return result



def compute_label_obj_accuracies(pred_label, label_all, objs_all):
    """
    返回：
      label_acc: {label: acc}
      obj_acc:   {obj: acc}
    """
    pred_label = np.asarray(pred_label)
    label_all  = np.asarray(label_all)
    objs_all   = np.asarray(objs_all)
    assert len(pred_label) == len(label_all) == len(objs_all), "输入长度不一致"

    correct = (pred_label == label_all).astype(np.float64)

    # 按 label
    label_acc = {}
    for lab in np.unique(label_all):
        m = (label_all == lab)
        grp = correct[m]
        label_acc[lab.item() if hasattr(lab, "item") else lab] = float(np.mean(grp)) if grp.size > 0 else 0.0

    # 按 obj
    obj_acc = {}
    for obj in np.unique(objs_all):
        m = (objs_all == obj)
        grp = correct[m]
        obj_acc[obj.item() if hasattr(obj, "item") else obj] = float(np.mean(grp)) if grp.size > 0 else 0.0

    return label_acc, obj_acc


class AlignEvaler(AlignTrain):
    def __init__(self, config):
        super().__init__(config, init_loaders=False)

    def evaluate_one_epoch(self, data_loader):
        with torch.no_grad():
            self.model_proj.eval()
            pred_label_list = np.array([])
            label_list = np.array([])
            objs_list = np.array([])
            for i, data_batch in enumerate(tqdm(data_loader,
                                                total=len(data_loader),
                                                desc="Evaluating",
                                                unit="batch")):
                img_a, img_b, img_c, label_a, label_b, label_c, objs = self.databatch2imgs(data_batch)
                e_a, e_q_loss_a, z_a = self.img2emb(img_a)
                e_b, e_q_loss_b, z_b = self.img2emb(img_b)
                e_c, e_q_loss_c, z_c = self.img2emb(img_c)

                # Calculate accuracy
                e_all = torch.cat((e_a, e_b, e_c), dim=0)
                label_all = np.concatenate((label_a, label_b, label_c))
                pred_label = self.embs2label(e_all)

                pred_label_list = np.concatenate((pred_label_list, pred_label))
                label_list = np.concatenate((label_list, label_all))
                objs_list = np.concatenate((objs_list, np.array(np.concatenate((objs, objs, objs)))))
            label_acc, obj_acc = compute_label_obj_accuracies(pred_label_list, label_list, objs_list)
            overall_acc = float(np.mean(pred_label_list == label_list)) if len(label_list) > 0 else 0.0
            return label_acc, obj_acc, overall_acc



# def test_1():
#     pred_label = [0, 1, 2, 2, 4, 0, 1, 2, 3, 0, 0, 1, 2, 1, 4, 0, 2, 2, 3, 4]
#     label_all = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
#     objs_all = ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'A', 'B', 'C', 'A', 'B']

#     label_acc, obj_acc = compute_label_obj_accuracies(pred_label, label_all, objs_all)
#     print("Label Accuracies:", label_acc)
#     print("Object Accuracies:", obj_acc)


if "__main__" == __name__:
    EXP_ROOT_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/exp')
    sys.path.append(EXP_ROOT_PATH)
    EXP_NAME_LIST = ["2026.1.17_10codes_iconic"]
    DATA_SAMPLES = 2048
    ACC_THRESHOLD = 0.4
    # 统一 dataloader
    print('Initializing online synthetic dataloaders...')
    # val_dataloader, ood_dataloader = init_test_online_synth_dataloaders(num_samples=DATA_SAMPLES)
    val_dataloader, ood_dataloader = init_test_online_synth_dataloaders_paired(
        num_samples=DATA_SAMPLES,
        obj_list_val=OBJ_LIST,      # 第一套 icon
        obj_list_ood=OBJ_LIST_2,    # 第二套 icon
        canvas_size=(224,224)
    )
    # 统计 val_dataloader 中 a+b+c 的总体分布
    val_counts_all, val_ratios_all = count_numeric_labels_per_epoch(val_dataloader, key="all")
    ood_counts_all, ood_ratios_all = count_numeric_labels_per_epoch(ood_dataloader, key="all")
    print("VAL Counts (all):", val_counts_all)
    print("OOD Counts (all):", ood_counts_all)

    for exp_name in EXP_NAME_LIST:
        exp_path = os.path.join(EXP_ROOT_PATH, exp_name)
        all_results_details_path = os.path.join(exp_path, "all_results_details.json")  # Need to run statistic.py first
        selected_sub_exps = select_exps_by_train_acc(all_results_details_path, ACC_THRESHOLD)
        num_selected = len(selected_sub_exps)
        print(f'Exp {exp_name}: {num_selected} sub-experiments selected with Train Acc > {ACC_THRESHOLD}')
        os.chdir(exp_path)
        sys.path.append(exp_path)
        print(f'Exp path: {exp_path}')
        t_config = __import__('config')
        reload(t_config)
        sys.path.pop()
        print(t_config.CONFIG)

        # Init evaluator
        evaluator = AlignEvaler(t_config.CONFIG)

        # Init statistic lists
        val_label_acc_dict, val_obj_acc_dict, ood_label_acc_dict, ood_obj_acc_dict = {}, {}, {}, {}
        val_overall_acc_dict, ood_overall_acc_dict = {}, {}

        for (key, value) in selected_sub_exps.items():
            seb_exp_num = int(key.split('_')[-1])
            ckpt_epoch = value
            print(f'  Sub-exp {seb_exp_num}, ckpt epoch {ckpt_epoch}')
            ckpt_path = os.path.join(exp_path, str(seb_exp_num), f'checkpoint_{ckpt_epoch}.pt')
            evaluator.load_model(ckpt_path)
            with torch.no_grad():
                # Evaluate on validation set
                val_label_acc_dict[key], val_obj_acc_dict[key], val_overall_acc_dict[key] = evaluator.evaluate_one_epoch(val_dataloader)
                # Evaluate on OOD set
                ood_label_acc_dict[key], ood_obj_acc_dict[key], ood_overall_acc_dict[key] = evaluator.evaluate_one_epoch(ood_dataloader)

        # Save results
        all_results = {
            "val_label_acc": val_label_acc_dict,
            "val_obj_acc": val_obj_acc_dict,
            "ood_label_acc": ood_label_acc_dict,
            "ood_obj_acc": ood_obj_acc_dict,
            "val_overall_acc": val_overall_acc_dict,
            "ood_overall_acc": ood_overall_acc_dict,
            "val_n_label_counts_all": val_counts_all,
            "val_n_label_ratios_all": val_ratios_all,
            "ood_n_label_counts_all": ood_counts_all,
            "ood_n_label_ratios_all": ood_ratios_all
        }
        results_path = os.path.join(exp_path, f'eval_results.json')
        upsert_json(results_path, all_results)
        print(f'Saved evaluation results to {results_path}')








