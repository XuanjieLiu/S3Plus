import sys
import os
import numpy as np
import json

from plot_multistyle_zc import MultiStyleZcEvaler

sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from interpolate_plus_accu_eval import InterpolatePlusAccuEval
from common_func import (load_config_from_exp_name, record_num_list, EXP_ROOT,
                         find_optimal_checkpoint_num_by_train_config, solve_label_emb_one2one_matching)
from eval_plus_nd import VQvaePlusEval, calc_one2one_plus_accu, calc_multi_emb_plus_accu, \
    calc_plus_z_self_cycle_consistency, calc_plus_z_mode_emb_label_cycle_consistency
from dataloader_plus import Dataset
from dataloader import SingleImgDataset, load_enc_eval_data_with_style
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from two_dim_num_vis import MumEval


EXP_NUM_LIST = [str(i) for i in range(1, 21)]
# EXP_NUM_LIST = ['1']
EXP_NAME_LIST = [
    # "2025.06.18_100vq_Zc[1]_Zs[0]_edim2_[0-20]_plus1024_1_tripleSet_Fullsymm_OnlineBlur",
    "2025.07.02_20vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_SingleStyleMahjong_nothing",
    "2025.07.02_20vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_SingleStyleMahjong_PureVQ",
    "2025.07.02_20vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_SingleStyleMahjong_symm",
    "2025.07.02_20vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_SingleStyleMahjong_trainAll",
    "2025.06.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyleMahjong_nothing",
    "2025.06.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyleMahjong_PureVQ",
    "2025.06.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyleMahjong_symm",
    "2025.06.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyleMahjong_trainAll",
]


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

    all_results_details = {'ckpts': [f'exp_{sub_exp}: {os.path.basename(ckpt)}' for sub_exp, ckpt in zip(EXP_NUM_LIST, all_ckpts)]}
    for key, value in all_results.items():
        all_results_details[key] = [f'exp_{sub_exp}: {round(v, 3)}' for sub_exp, v in zip(EXP_NUM_LIST, value)]
    details_path = os.path.join(pipline_dir, 'all_results_details.json')
    with open(details_path, 'w') as f:
        json.dump(all_results_details, f, indent=4)


if __name__ == "__main__":
    for exp_name in EXP_NAME_LIST:
        exp_path = os.path.join(EXP_ROOT, exp_name)
        config = load_config_from_exp_name(exp_name)
        eval_config = config['eval_config']
        pipeline_dir = os.path.join(exp_path, eval_config['pipeline_result_path'])
        os.makedirs(pipeline_dir, exist_ok=True)
        all_results = {}
        all_ckpts = find_all_ckpts(config, exp_path)

        # eval plus accu
        if eval_config.get('plus_eval_configs') is not None:
            plus_evaler = VQvaePlusEval(config)
            plus_eval_configs = eval_config['plus_eval_configs']
            for sub_config in plus_eval_configs:
                name = sub_config['name']
                plus_eval_set_path_list = sub_config['eval_set_path_list']
                # 如果有多个 set_path, 创建多个 dataset 并合并
                datasets = ConcatDataset([Dataset(path) for path in plus_eval_set_path_list])
                data_loader = DataLoader(datasets, batch_size=config['batch_size'], shuffle=False)
                one2n_accu_result_name = f"{name}_one2n_accu"
                one2one_accu_result_name = f"{name}_one2one_accu"
                one2n_accu_result_name_cycle = f"{name}_one2n_accu_cycle"
                one2one_accu_result_name_cycle = f"{name}_one2one_accu_cycle"
                emb_self_consistency_result_name = f"{name}_emb_self_consistency"
                emb_label_consistency_result_name = f"{name}_emb_label_consistency"
                z_c_recognition_rate_result_name = f"{name}_z_c_recognition_rate"
                z_c_cycle_recognition_rate_result_name = f"{name}_z_c_cycle_recognition_rate"
                one2n_accu_list = []
                one2n_accu_list_cycle = []
                one2one_accu_list = []
                one2one_accu_list_cycle = []
                emb_self_consistency_list = []
                emb_label_consistency_list = []
                z_c_recognition_rate_list = []
                z_c_cycle_recognition_rate_list = []
                for sub_exp, ckpt_path in zip(EXP_NUM_LIST, all_ckpts):
                    plus_evaler.reload_model(ckpt_path)
                    all_enc_z, all_plus_z = plus_evaler.load_plusZ_eval_data(data_loader, is_find_index=True)
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
                all_results[emb_self_consistency_result_name] = emb_self_consistency_list
                all_results[emb_label_consistency_result_name] = emb_label_consistency_list
                all_results[z_c_recognition_rate_result_name] = z_c_recognition_rate_list
                all_results[z_c_cycle_recognition_rate_result_name] = z_c_cycle_recognition_rate_list

        # eval matching rate
        if eval_config.get('emb_matching_rate_configs') is not None:
            mr_evaler = MultiStyleZcEvaler(config)
            emb_matching_rate_configs = eval_config['emb_matching_rate_configs']
            for sub_config in emb_matching_rate_configs:
                name = sub_config['name']
                eval_set_path_list = sub_config['eval_set_path_list']
                # 如果有多个 set_path, 创建多个 dataset 并合并
                datasets = ConcatDataset([SingleImgDataset(path) for path in eval_set_path_list])
                data_loader = DataLoader(datasets, batch_size=config['batch_size'], shuffle=False)
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
        if eval_config.get('orderliness_configs') is not None:
            orderliness_evaler = MumEval(config, None)
            orderliness_configs = eval_config['orderliness_configs']
            for sub_config in orderliness_configs:
                name = sub_config['name']
                img_dir_path = os.path.join(pipeline_dir, sub_config['img_dir_name'])
                os.makedirs(img_dir_path, exist_ok=True)
                eval_set_path_list = sub_config['eval_set_path_list']
                is_add_noise = sub_config.get('is_add_noise', False)
                # 如果有多个 set_path, 创建多个 dataset 并合并
                datasets = ConcatDataset([SingleImgDataset(path) for path in eval_set_path_list])
                data_loader = DataLoader(datasets, batch_size=config['batch_size'], shuffle=False)
                orderliness_list = []
                for sub_exp, ckpt_path in zip(EXP_NUM_LIST, all_ckpts):
                    img_path = os.path.join(img_dir_path, f'{sub_exp}.png')
                    orderliness_evaler.reload_model(ckpt_path)
                    if is_add_noise:
                        nna_score = orderliness_evaler.num_eval_two_dim_with_gaussian_noise(
                            data_loader, img_path, noise_batch=1
                        )
                    else:
                        nna_score = orderliness_evaler.num_eval_two_dim(data_loader, img_path)
                    orderliness_list.append(nna_score)
                all_results[f'{name}'] = orderliness_list

        # save all_results to json
        save_all_results(pipeline_dir, all_results, all_ckpts)

