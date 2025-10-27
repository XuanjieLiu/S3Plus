import sys
import os
from typing import Dict, Callable, Any
import torch
import numpy as np
import json
from torchvision import transforms
from plot_multistyle_zc import MultiStyleZcEvaler

sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from common_func import (load_config_from_exp_name, record_num_list, EXP_ROOT, RandomGaussianBlur, make_dataset_trans,
                         find_optimal_checkpoint_num_by_train_config, solve_label_emb_one2one_matching)
from eval_plus_nd import (VQvaePlusEval, calc_one2one_plus_accu, calc_multi_emb_plus_accu, calc_emb_select_plus_accu,
                          calc_plus_z_self_cycle_consistency, calc_plus_z_mode_emb_label_cycle_consistency,
                          interpolate_plus_eval, calc_emb_select_plus_accu_debug)
from dataloader_plus import MultiImgDataset
from dataloader import SingleImgDataset, load_enc_eval_data_with_style
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from two_dim_num_vis import MumEval

EVAL_ITEM_PLUS = 'plus_eval_configs'
EVAL_ITEM_MATCHING_RATE = 'emb_matching_rate_configs'
EVAL_ITEM_ORDERLINESS = 'orderliness_configs'
EVAL_ITEM_INTERPOLATE = 'interpolate_configs'

EXP_NUM_LIST = [str(i) for i in range(1, 21)]
# EXP_NUM_LIST = ['1']
EXP_NAME_LIST = [
    # High-dim test blue points orderliness
    "2025.10.26_10vq_Zc[4]_Zs[0]_edim4_[0-20]_plus1024_1_tripleSet_Fullsymm",
    "2025.10.26_16vq_Zc[8]_Zs[0]_edim8_[0-20]_plus1024_1_tripleSet_Fullsymm",

    # High-dim test Mahjong
    # "2025.09.27_10vq_Zc[4]_Zs[0]_edim1_[0-20]_plus1024_1_SingleStyleMahjong_symm",
    # "2025.09.27_256vq_Zc[1]_Zs[0]_edim1_[0-20]_plus1024_1_SingleStyleMahjong_symm",
    # "2025.09.28_10vq_Zc[4]_Zs[0]_edim2_[0-20]_plus1024_1_SingleStyleMahjong_symm",
    # "2025.09.28_16vq_Zc[4]_Zs[0]_edim1_[0-20]_plus1024_1_SingleStyleMahjong_symm",
    # "2025.09.29_10vq_Zc[4]_Zs[0]_edim4_[0-20]_plus1024_1_SingleStyleMahjong_symm",

    # # continue Mahjong
    # "2025.08.23_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyleMahjong_symm_continue",
    # "2025.08.23_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyleMahjong_trainAll_continue",

    # # Main experiments, single style blue points with blur
    # "2025.06.13_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Fullsymm_OnlineBlur",
    # "2025.06.16_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Nothing_OnlineBlur",
    # "2025.06.13_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Nothing_trainAll_OnlineBlur",
    # "2025.06.16_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_PureVQ_OnlineBlur",
    # "2025.06.18_100vq_Zc[1]_Zs[0]_edim2_[0-20]_plus1024_1_tripleSet_Fullsymm_OnlineBlur",
    # "2025.06.18_100vq_Zc[1]_Zs[0]_edim2_[0-20]_plus1024_1_tripleSet_Nothing_trainAll_OnlineBlur",

    # # Main experiments, multi style colourful points
    # "2025.05.15_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_Fullsymm",
    # "2025.05.15_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_Nothing",
    # "2025.05.19_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_Nothing_trainAll",
    # "2025.06.10_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_PureVQ",
    #
    # Main experiments, multi style mahjong
    # "2025.06.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyleMahjong_nothing",
    # "2025.06.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyleMahjong_PureVQ",
    # "2025.06.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyleMahjong_symm",
    # "2025.06.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyleMahjong_trainAll",
    #
    # # Main experiments, single style blue points
    # "2025.05.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Fullsymm",
    # "2025.05.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Nothing",
    # "2025.05.19_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Nothing_trainAll",
    # "2025.06.05_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_PureVQ",

    # # single style mahjong, pending
    # "2025.07.02_20vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_SingleStyleMahjong_nothing",
    # "2025.07.02_20vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_SingleStyleMahjong_PureVQ",
    # "2025.07.02_20vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_SingleStyleMahjong_symm",
    # "2025.07.02_20vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_SingleStyleMahjong_trainAll",
]
EVAL_TERMS = [
    EVAL_ITEM_PLUS,
    EVAL_ITEM_MATCHING_RATE,
    EVAL_ITEM_ORDERLINESS,
    EVAL_ITEM_INTERPOLATE,
]


def make_data_loader(sub_cfg: Dict[str, Any],
                     dataset_cls: Callable[..., Any]) -> DataLoader:
    aug_t = sub_cfg.get('augment_times', 1)
    path_list = sub_cfg['eval_set_path_list']
    is_blur = sub_cfg.get('is_blur', False)
    blur_cfg = sub_cfg.get('blur_config', {})
    trans = make_dataset_trans(is_blur, blur_cfg)
    # 如果有多个 set_path, 创建多个 dataset 并合并
    datasets = ConcatDataset([dataset_cls(path, augment_times=aug_t, transform=trans)
                              for path in path_list])
    data_loader = DataLoader(datasets, batch_size=128, shuffle=False)
    return data_loader


def find_ckpt(config, exp_path, sub_exp):
    eval_config = config['eval_config']
    optimal_checkpoint_finding_config = eval_config['optimal_checkpoint_finding_config']
    sub_exp_path = os.path.join(exp_path, sub_exp)
    optimal_checkpoint_num = find_optimal_checkpoint_num_by_train_config(sub_exp_path, config,
                                                                         optimal_checkpoint_finding_config)
    print(f'Optimal checkpoint number for {sub_exp}: {optimal_checkpoint_num}')
    checkpoint_path = os.path.join(exp_path, sub_exp, f'checkpoint_{optimal_checkpoint_num}.pt')
    return checkpoint_path


def find_all_ckpts(config, exp_path):
    ckpt_list = []
    for sub_exp in EXP_NUM_LIST:
        checkpoint_path = find_ckpt(config, exp_path, sub_exp)
        ckpt_list.append(checkpoint_path)
    return ckpt_list


def save_all_results(pipline_dir, all_results, all_ckpts):
    # save all_results to json
    result_path = os.path.join(pipline_dir, 'all_results.json')
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    all_results_summary = {}
    for key, value in all_results.items():
        all_results_summary[key] = {
            'value': f'{round(np.mean(value), 2)} \\pm {round(np.std(value), 2)}',
            'mean': np.mean(value),
            'std': np.std(value),
            'max': np.max(value),
            'min': np.min(value),
            'count': len(value),
        }
    summary_path = os.path.join(pipline_dir, 'all_results_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results_summary, f, indent=4)

    all_results_details = {
        'ckpts': [f'exp_{sub_exp}: {os.path.basename(ckpt)}' for sub_exp, ckpt in zip(EXP_NUM_LIST, all_ckpts)]}
    for key, value in all_results.items():
        all_results_details[key] = [f'exp_{sub_exp}: {round(v, 3)}' for sub_exp, v in zip(EXP_NUM_LIST, value)]
    details_path = os.path.join(pipline_dir, 'all_results_details.json')
    with open(details_path, 'w') as f:
        json.dump(all_results_details, f, indent=4)


def pipeline_eval(exp_name: str):
    exp_path = os.path.join(EXP_ROOT, exp_name)
    config = load_config_from_exp_name(exp_name)
    eval_config = config['eval_config']
    pipeline_dir = os.path.join(exp_path, eval_config['pipeline_result_path'])
    os.makedirs(pipeline_dir, exist_ok=True)
    all_results = {}
    all_ckpts = find_all_ckpts(config, exp_path)
    # eval plus accu
    if eval_config.get(EVAL_ITEM_PLUS) is not None and EVAL_ITEM_PLUS in EVAL_TERMS:
        plus_evaler = VQvaePlusEval(config)
        plus_eval_configs = eval_config['plus_eval_configs']
        for sub_config in plus_eval_configs:
            name = sub_config['name']
            data_loader = make_data_loader(sub_config, MultiImgDataset)
            one2n_accu_result_name = f"{name}_one2n_accu"
            one2one_accu_result_name = f"{name}_one2one_accu"
            one2n_accu_result_name_cycle = f"{name}_one2n_accu_cycle"
            one2one_accu_result_name_cycle = f"{name}_one2one_accu_cycle"
            emb_select_accu_result_name = f"{name}_emb_select_accu"
            emb_select_accu_cycle_result_name = f"{name}_emb_select_accu_cycle"
            emb_self_consistency_result_name = f"{name}_emb_self_consistency"
            emb_label_consistency_result_name = f"{name}_emb_label_consistency"
            z_c_recognition_rate_result_name = f"{name}_z_c_recognition_rate"
            z_c_cycle_recognition_rate_result_name = f"{name}_z_c_cycle_recognition_rate"
            one2n_accu_list = []
            one2n_accu_list_cycle = []
            one2one_accu_list = []
            one2one_accu_list_cycle = []
            emb_select_accu_list = []
            emb_select_accu_cycle_list = []
            emb_self_consistency_list = []
            emb_label_consistency_list = []
            z_c_recognition_rate_list = []
            z_c_cycle_recognition_rate_list = []
            for sub_exp, ckpt_path in zip(EXP_NUM_LIST, all_ckpts):
                plus_evaler.reload_model(ckpt_path)
                all_enc_z, all_plus_z = plus_evaler.load_plusZ_eval_data(data_loader)

                # # debug emb select accu
                # debug_save_dir = os.path.join(pipeline_dir, f'debug_emb_select_accu_{sub_exp}')
                # os.makedirs(debug_save_dir, exist_ok=True)
                # emb_select_accu, emb_select_accu_cycle = calc_emb_select_plus_accu_debug(all_enc_z, all_plus_z, plus_evaler.model, debug_save_dir)
                # emb_select_accu_list.append(emb_select_accu)
                # emb_select_accu_cycle_list.append(emb_select_accu_cycle)

                # 计算 emb select accu
                emb_select_accu, emb_select_accu_cycle = calc_emb_select_plus_accu(all_enc_z, all_plus_z)
                emb_select_accu_list.append(emb_select_accu)
                emb_select_accu_cycle_list.append(emb_select_accu_cycle)
                # 计算 one2n accu
                one2n_accu, one2n_accu_cycle = calc_multi_emb_plus_accu(all_enc_z, all_plus_z)
                one2n_accu_list.append(one2n_accu)
                one2n_accu_list_cycle.append(one2n_accu_cycle)
                # 计算 one2one accu
                one2one_accu, one2one_accu_cycle = calc_one2one_plus_accu(all_enc_z, all_plus_z)
                one2one_accu_list.append(one2one_accu)
                one2one_accu_list_cycle.append(one2one_accu_cycle)
                # 计算 emb self consistency
                emb_self_consistency_list.append(calc_plus_z_self_cycle_consistency(all_plus_z))
                # 计算 emb label consistency 和 z_c recognition rate
                emb_label_consistency, z_c_recognition_rate, z_c_cycle_recognition_rate = (
                    calc_plus_z_mode_emb_label_cycle_consistency(all_enc_z, all_plus_z))
                emb_label_consistency_list.append(emb_label_consistency)
                z_c_recognition_rate_list.append(z_c_recognition_rate)
                z_c_cycle_recognition_rate_list.append(z_c_cycle_recognition_rate)
            all_results[one2n_accu_result_name] = one2n_accu_list
            all_results[one2one_accu_result_name] = one2one_accu_list
            all_results[one2n_accu_result_name_cycle] = one2n_accu_list_cycle
            all_results[one2one_accu_result_name_cycle] = one2one_accu_list_cycle
            all_results[emb_select_accu_result_name] = emb_select_accu_list
            all_results[emb_select_accu_cycle_result_name] = emb_select_accu_cycle_list
            all_results[emb_self_consistency_result_name] = emb_self_consistency_list
            all_results[emb_label_consistency_result_name] = emb_label_consistency_list
            all_results[z_c_recognition_rate_result_name] = z_c_recognition_rate_list
            all_results[z_c_cycle_recognition_rate_result_name] = z_c_cycle_recognition_rate_list

    # eval matching rate
    if eval_config.get(EVAL_ITEM_MATCHING_RATE) is not None and EVAL_ITEM_MATCHING_RATE in EVAL_TERMS:
        mr_evaler = MultiStyleZcEvaler(config)
        emb_matching_rate_configs = eval_config['emb_matching_rate_configs']
        for sub_config in emb_matching_rate_configs:
            name = sub_config['name']
            data_loader = make_data_loader(sub_config, SingleImgDataset)
            one2n_matching_rate_list = []
            one2one_matching_rate_list = []
            for sub_exp, ckpt_path in zip(EXP_NUM_LIST, all_ckpts):
                mr_evaler.reload_model(ckpt_path)
                num_z, num_labels, colors, shapes = load_enc_eval_data_with_style(
                    data_loader,
                    lambda x: mr_evaler.model.find_indices(
                        mr_evaler.model.batch_encode_to_z(x)[0],
                        True, False
                    )
                )
                num_emb_idx = [x[0] for x in num_z.detach().cpu().numpy()]
                one2n_match_rate = mr_evaler.calc_emb_matching_score(num_emb_idx, num_labels)
                one2n_matching_rate_list.append(one2n_match_rate)
                one2one_matching_rate = solve_label_emb_one2one_matching(num_emb_idx, num_labels)[1]
                one2one_matching_rate_list.append(one2one_matching_rate)
            all_results[f'{name}_one2n'] = one2n_matching_rate_list
            all_results[f'{name}_one2one'] = one2one_matching_rate_list

    # eval orderliness
    if eval_config.get(EVAL_ITEM_ORDERLINESS) is not None and EVAL_ITEM_ORDERLINESS in EVAL_TERMS:
        orderliness_evaler = MumEval(config, None)
        orderliness_configs = eval_config['orderliness_configs']
        for sub_config in orderliness_configs:
            name = sub_config['name']
            img_dir_path = os.path.join(pipeline_dir, sub_config['img_dir_name'])
            os.makedirs(img_dir_path, exist_ok=True)
            data_loader = make_data_loader(sub_config, SingleImgDataset)
            orderliness_list = []
            for sub_exp, ckpt_path in zip(EXP_NUM_LIST, all_ckpts):
                img_path = os.path.join(img_dir_path, f'{sub_exp}.png')
                orderliness_evaler.reload_model(ckpt_path)
                nna_score = orderliness_evaler.num_eval_two_dim(data_loader, img_path)
                orderliness_list.append(nna_score)
            all_results[f'{name}'] = orderliness_list

    # eval interpolate
    if eval_config.get(EVAL_ITEM_INTERPOLATE) is not None and EVAL_ITEM_INTERPOLATE in EVAL_TERMS:
        interpolate_configs = eval_config['interpolate_configs']
        interpolate_evaler = VQvaePlusEval(config)
        for sub_config in interpolate_configs:
            name = sub_config['name']
            data_loader = make_data_loader(sub_config, MultiImgDataset)
            itp_result_name = f"{name}"
            itp_cycle_result_name = f"{name}_cycle"
            itp_result_list = []
            itp_cycle_result_list = []
            for sub_exp, ckpt_path in zip(EXP_NUM_LIST, all_ckpts):
                interpolate_evaler.reload_model(ckpt_path)
                all_enc_z, all_plus_z = interpolate_evaler.load_plusZ_eval_data(data_loader)
                # 计算 interpolate
                itp_score, itp_cycle_score = interpolate_plus_eval(interpolate_evaler.model,
                                                                   all_enc_z, all_plus_z,
                                                                   sub_config.get('interpolate_num', 10))
                itp_result_list.append(itp_score)
                itp_cycle_result_list.append(itp_cycle_score)
            all_results[itp_result_name] = itp_result_list
            all_results[itp_cycle_result_name] = itp_cycle_result_list

    # save all_results to json
    save_all_results(pipeline_dir, all_results, all_ckpts)


if __name__ == "__main__":
    with torch.no_grad():
        for exp_name in EXP_NAME_LIST:
            pipeline_eval(exp_name)
        print("Pipeline evaluation completed.")
