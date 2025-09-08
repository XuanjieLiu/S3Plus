import sys
import os
import json
import numpy as np
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from VQ.common_func import load_config_from_exp_name, record_num_list
from statistic import find_best_epoch, get_metrics_by_epoch
from loss_counter import read_record
from dataloader_plus import MultiImgDataset

EXP_ROOT = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/exp')
EXP_NUM_LIST = [str(i) for i in range(1, 21)]
EXP_NAME_LIST = [
    "2025.9.05_10codes",
]
EVAL_RECORD_NAMES = [
    'Train_record.txt',
    'Val_record.txt',
    'Val_ood_record.txt',
]

EVAL_KEYS = ['accuracy']
TARGET_KEY = 'accuracy'

STATISTIC_RESULT_PATH = 'statistic_results.txt'

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

def batch_statistic():
    for exp_name in EXP_NAME_LIST:
        all_results = {}
        all_ckpts = []
        exp_path = os.path.join(EXP_ROOT, exp_name)
        config = load_config_from_exp_name(exp_name, exp_root=EXP_ROOT, config_name='config')
        for sub_exp in EXP_NUM_LIST:
            sub_exp_path = os.path.join(exp_path, sub_exp)
            train_record_path = os.path.join(sub_exp_path, config['ALIGN']['train_record_path'])
            optimal_checkpoint_num = find_best_epoch(train_record_path, TARGET_KEY, True)
            all_ckpts.append(f"{optimal_checkpoint_num}")
            for eval_record_name in EVAL_RECORD_NAMES:
                eval_txt_path = os.path.join(sub_exp_path, eval_record_name)
                eval_record = get_metrics_by_epoch(eval_txt_path, optimal_checkpoint_num, EVAL_KEYS)
                for i in range(len(EVAL_KEYS)):
                    key = EVAL_KEYS[i]
                    result_name = f"{eval_record_name.split('.')[0]}_{key}"
                    if result_name not in all_results:
                        all_results[result_name] = []
                    all_results[result_name].append(eval_record[i])
            # save all_results to json
        save_all_results(exp_path, all_results, all_ckpts)








if __name__ == "__main__":
    batch_statistic()

