from train import PlusTrainer
from common_func import DATASET_ROOT, EXP_ROOT, load_config_from_exp_name
from VQVAE import VQVAE
import torch
import os
import json


class FewShotEvaler(PlusTrainer):
    def __init__(self, config, model_path=None, is_fix_vq=True):
        super().__init__(config, True)
        if model_path is not None:
            self.load_model(model_path)
            print(f"Model is loaded from {model_path}")
        self.optimizer = None

        cancel_grad(self.model.plus_net)
        optim_param_list = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        if is_fix_vq:
            cancel_grad(self.model.vq_layer)
        else:
            optim_param_list += list(self.model.vq_layer.parameters())
        self.optimizer = torch.optim.Adam(optim_param_list, lr=self.learning_rate)

    def load_model(self, model_path):
        self.model.load_state_dict(self.model.load_tensor(model_path))

    def train(self, is_resume=True, optimizer: torch.optim.Optimizer = None):
        super().train(is_resume, optimizer=self.optimizer)


def cancel_grad(model):
    for param in model.parameters():
        param.requires_grad = False


EVAL_CONFIG = {
    'is_fix_vq': True,
    'learning_rate': 1e-5,
    'is_normal_train': False,
}

BASE_FEW_SHOT_CONFIG = {
    'max_iter_num': 1001,
    'log_interval': 20,
    'batch_size': 2048,
    'checkpoint_interval': 200,
    'learning_rate': EVAL_CONFIG['learning_rate'],
}

NORMAL_TRAIN_CONFIG = {
    'is_commutative_train': False,
    'is_commutative_all': False,
    'commutative_z_loss_scalar': 0.0,
    'associative_z_loss_scalar': 0.0,
}

FEW_SHOT_EXP = [
    {
        'name': 'newColor',
        'result_path': 'newColor_fewShot_eval/',
        'config': {
            'train_data_path': f"{DATASET_ROOT}/multi_style_realPairs_plus_eval_newColor/shot_16_test",
            'single_img_eval_set_path': f"{DATASET_ROOT}/multi_style_eval_(0,20)_FixedPos_newColor",
            'plus_eval_set_path': f"{DATASET_ROOT}/multi_style_realPairs_plus_eval_newColor/train",
        }
    },
    # {
    #     'name': 'newShape',
    #     'result_path': 'newShape_fewShot_eval',
    #     'config': {
    #         'train_data_path': f"{DATASET_ROOT}/multi_style_realPairs_plus_eval_newShape/shot_16_test",
    #         'single_img_eval_set_path': f"{DATASET_ROOT}/multi_style_eval_(0,20)_FixedPos_newShape",
    #         'plus_eval_set_path': f"{DATASET_ROOT}/multi_style_realPairs_plus_eval_newShape/train",
    #     }
    # },
    # {
    #     'name': 'newShapeColor',
    #     'result_path': 'newShapeColor_fewShot_eval',
    #     'config': {
    #         'train_data_path': f"{DATASET_ROOT}/multi_style_realPairs_plus_eval_newShapeColor/shot_16_test",
    #         'single_img_eval_set_path': f"{DATASET_ROOT}/multi_style_eval_(0,20)_FixedPos_newShapeColor",
    #         'plus_eval_set_path': f"{DATASET_ROOT}/multi_style_realPairs_plus_eval_newShapeColor/train",
    #     }
    # }
]

RESULT_ROOT = f'few_shot_eval_result_lr-{EVAL_CONFIG["learning_rate"]}_fixVQ-{EVAL_CONFIG["is_fix_vq"]}_normalTrain-{EVAL_CONFIG["is_normal_train"]}'



def eval_an_sub_exp(exp_name: str, sub_exp, check_point_name, eval_config=None, result_root=RESULT_ROOT):
    if eval_config is None:
        eval_config = EVAL_CONFIG
    sub_exp_path = os.path.join(EXP_ROOT, exp_name, str(sub_exp))
    os.chdir(sub_exp_path)
    config = load_config_from_exp_name(exp_name)
    check_point_path = os.path.join(sub_exp_path, check_point_name)
    eval_dir = os.path.join(sub_exp_path, result_root)
    os.makedirs(eval_dir, exist_ok=True)
    os.chdir(eval_dir)
    formatted_eval_config = json.dumps(eval_config, indent=4)
    # 将格式化的 JSON 字符串写入 txt 文件
    with open('eval_config.txt', 'w') as file:
        file.write(formatted_eval_config)
    for i in range(len(FEW_SHOT_EXP)):
        few_shot_config = {**config, **BASE_FEW_SHOT_CONFIG, **FEW_SHOT_EXP[i]['config']}
        if eval_config['is_normal_train']:
            few_shot_config = {**few_shot_config, **NORMAL_TRAIN_CONFIG}
        few_shot_exp_path = os.path.join(eval_dir, FEW_SHOT_EXP[i]['result_path'])
        os.makedirs(few_shot_exp_path, exist_ok=True)
        os.chdir(few_shot_exp_path)
        evaler = FewShotEvaler(few_shot_config, check_point_path, is_fix_vq=eval_config['is_fix_vq'])
        evaler.load_model(check_point_path)
        evaler.train(is_resume=True)
        print(f"Few shot evaluation done for {FEW_SHOT_EXP[i]['name']}")



if __name__ == "__main__":
    EXP_NAME = '2024.04.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_AssocFullsymmCommu'
    SUB_EXP = 2
    CHECK_POINT = 'checkpoint_10000.pt'
    eval_an_sub_exp(EXP_NAME, SUB_EXP, CHECK_POINT, eval_config=EVAL_CONFIG, result_root=RESULT_ROOT)

    EXP_NAME = '2024.04.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_Nothing'
    SUB_EXP = 6
    CHECK_POINT = 'checkpoint_9500.pt'
    eval_an_sub_exp(EXP_NAME, SUB_EXP, CHECK_POINT, eval_config=EVAL_CONFIG, result_root=RESULT_ROOT)

