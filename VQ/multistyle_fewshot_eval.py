from train import PlusTrainer
from common_func import DATASET_ROOT, EXP_ROOT, load_config_from_exp_name
from dataMaker_newStyle_from_knownPairs import load_data_pairs_from_dataset
from dataloader_plus import MultiImgDataset
from dataloader import SingleImgDataset, load_enc_eval_data, load_enc_eval_data_with_style
from VQVAE import VQVAE
import torch
import os
import json
from torch.utils.data import random_split, DataLoader, ConcatDataset
import wandb
from eval_multi_style import find_mode_label_zc, determine_accu_by_mode, eval_accu, load_idx_and_label



# This is secret and shouldn't be checked into version control
WANDB_API_KEY="532007cd7a07c1aa0d1194049c3231dadd1d418e"
# Name and notes optional
wandb.login(key=WANDB_API_KEY)

class FewShotEvaler(PlusTrainer):
    def __init__(self, config, model_path=None, is_fix_vq=True):
        super().__init__(config, True)
        if model_path is not None:
            self.load_model(model_path)
            print(f"Model is loaded from {model_path}")
        self.optimizer = None

        cancel_grad(self.model.plus_net)
        if is_fix_vq:
            cancel_grad(self.model.vq_layer)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        original_train_set_path = config['train_data_path']
        original_train_set = MultiImgDataset(original_train_set_path)
        self.original_train_loader = DataLoader(original_train_set, batch_size=self.batch_size)

        comb_old_set, _ = split_train_eval(original_train_set_path, 0.1, random_seed=41)
        few_shot_train_data_path = config['few_shot_train_data_path']
        train_set, eval_set = split_train_eval(few_shot_train_data_path, config['train_ratio'])
        merged_train_set = ConcatDataset([comb_old_set, train_set]) if config['mix_old_train'] else train_set
        self.loader = DataLoader(merged_train_set, batch_size=self.batch_size, shuffle=True)
        self.plus_eval_loader = DataLoader(eval_set, batch_size=self.batch_size)

        few_shot_single_img_eval_set_path = config['few_shot_single_img_eval_set_path']
        few_shot_single_img_eval_set = SingleImgDataset(few_shot_single_img_eval_set_path)
        self.single_img_eval_loader = DataLoader(few_shot_single_img_eval_set, batch_size=self.batch_size)

        original_single_img_eval_set = SingleImgDataset(f"{DATASET_ROOT}multi_style_eval_(0,20)_FixedPos_TrainStyle")
        self.original_single_img_eval_loader = DataLoader(original_single_img_eval_set, batch_size=self.batch_size)

        original_idx, original_labels = load_idx_and_label(self.original_single_img_eval_loader, self.model)
        self.original_mode_dict = find_mode_label_zc(original_idx, original_labels)

    def load_model(self, model_path):
        self.model.load_state_dict(self.model.load_tensor(model_path))

    def train(self, is_resume=True, optimizer: torch.optim.Optimizer = None, eval_func: callable = None):
        super().train(is_resume, optimizer=self.optimizer, eval_func=self.addition_eval_func)

    def addition_eval_func(self, epoch, model):
        if epoch % self.log_interval == 0:
            original_idx, original_labels = load_idx_and_label(self.original_single_img_eval_loader, self.model)
            few_shot_eval_idx, few_shot_eval_labels = load_idx_and_label(self.single_img_eval_loader, self.model)
            new_match_accu = eval_accu(original_idx, original_labels, few_shot_eval_idx, few_shot_eval_labels)
            original_match_accu = determine_accu_by_mode(self.original_mode_dict, original_idx, original_labels)

            ood_train_ks, ood_train_accu = self.calc_plus_accu(self.loader)
            ood_eval_ks, ood_eval_accu = self.calc_plus_accu(self.plus_eval_loader)
            origin_train_ks, origin_train_accu = self.calc_plus_accu(self.original_train_loader)

            wandb.log({
                'new_match_accu': new_match_accu,
                'original_match_accu': original_match_accu,
                'ood_train_plus_accu': ood_train_accu,
                'ood_eval_plus_accu': ood_eval_accu,
                'origin_train_plus_accu': origin_train_accu,
                'epoch': epoch
            })





def cancel_grad(model):
    for param in model.parameters():
        param.requires_grad = False



EVAL_CONFIG = {
    'is_fix_vq': True,
    'learning_rate': 1e-4,
    'is_normal_train': False,
    'train_ratio': 0.1,
    'mix_old_train': False,
}

BASE_FEW_SHOT_CONFIG = {
    'max_iter_num': 801,
    'log_interval': 20,
    'batch_size': 2048,
    'checkpoint_interval': 200,
    'learning_rate': EVAL_CONFIG['learning_rate'],
    'train_ratio': EVAL_CONFIG['train_ratio'],
    'mix_old_train': EVAL_CONFIG['mix_old_train'],
}

NORMAL_TRAIN_CONFIG = {
    'is_commutative_train': False,
    'is_commutative_all': False,
    'commutative_z_loss_scalar': 0.0,
    'associative_z_loss_scalar': 0.0,
}

FEW_SHOT_EXP = [
    {
        'name': 'seenPair',
        'result_path': 'newShapeColor_fewShot_eval',
        'config': {
            'few_shot_train_data_path': f"{DATASET_ROOT}/multi_style_realPairs_plus_eval_newShapeColor/train",
            'few_shot_single_img_eval_set_path': f"{DATASET_ROOT}/multi_style_eval_(0,20)_FixedPos_newShapeColor",
        }
    },
    {
        'name': 'unseenPair',
        'result_path': 'newShapeColor_fewShot_eval',
        'config': {
            'few_shot_train_data_path': f"{DATASET_ROOT}/multi_style_realPairs_plus_eval_newShapeColor/test",
            'few_shot_single_img_eval_set_path': f"{DATASET_ROOT}/multi_style_eval_(0,20)_FixedPos_newShapeColor",
        }
    }
]

RESULT_ROOT = f'few_shot_eval_result_lr-{EVAL_CONFIG["learning_rate"]}_fixVQ-{EVAL_CONFIG["is_fix_vq"]}_normalTrain-{EVAL_CONFIG["is_normal_train"]}_trainRatio-{EVAL_CONFIG["train_ratio"]}'

def split_train_eval(data_path, train_ratio, random_seed=42):
    dataset = MultiImgDataset(data_path)
    n_train = int(len(dataset) * train_ratio)
    n_eval = len(dataset) - n_train
    # 创建一个生成器并设置种子
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, eval_dataset = random_split(dataset, [n_train, n_eval], generator=generator)
    print(len(train_dataset), len(eval_dataset))
    return train_dataset, eval_dataset

def eval_an_sub_exp(exp_name: str, sub_exp, check_point_name, eval_config=None, result_root=RESULT_ROOT):
    if eval_config is None:
        eval_config = EVAL_CONFIG
    sub_exp_path = os.path.join(EXP_ROOT, exp_name, str(sub_exp))
    os.chdir(sub_exp_path)
    config = load_config_from_exp_name(exp_name)
    check_point_path = os.path.join(sub_exp_path, check_point_name)
    formatted_eval_config = json.dumps(eval_config, indent=4)
    # 将格式化的 JSON 字符串写入 txt 文件
    with open('eval_config.txt', 'w') as file:
        file.write(formatted_eval_config)
    for i in range(len(FEW_SHOT_EXP)):
        eval_dir = os.path.join(sub_exp_path, f'{result_root}_{FEW_SHOT_EXP[i]["name"]}')
        os.makedirs(eval_dir, exist_ok=True)
        os.chdir(eval_dir)
        few_shot_config = {**config, **BASE_FEW_SHOT_CONFIG, **FEW_SHOT_EXP[i]['config']}
        if eval_config['is_normal_train']:
            few_shot_config = {**few_shot_config, **NORMAL_TRAIN_CONFIG}
        wandb.init(
            project="few_shot_eval",
            name=f"{exp_name.split('_')[-1]}_{sub_exp}",
            tags=[
                  f"train_ratio={EVAL_CONFIG['train_ratio']}",
                  f'fixVQ={EVAL_CONFIG["is_fix_vq"]}',
                  f'normalTrain={EVAL_CONFIG["is_normal_train"]}',
                  f'checkPoint={check_point_name}'
                  f'lr={EVAL_CONFIG["learning_rate"]}',
                  f'mix_old_train={EVAL_CONFIG["mix_old_train"]}',
                  ],
            group=f"{FEW_SHOT_EXP[i]['name']}",
            config=few_shot_config
        )
        few_shot_exp_path = os.path.join(eval_dir, FEW_SHOT_EXP[i]['result_path'])
        os.makedirs(few_shot_exp_path, exist_ok=True)
        os.chdir(few_shot_exp_path)
        evaler = FewShotEvaler(few_shot_config, check_point_path, is_fix_vq=eval_config['is_fix_vq'])
        evaler.load_model(check_point_path)
        evaler.train(is_resume=True)
        wandb.finish()
        print(f"Few shot evaluation done for {FEW_SHOT_EXP[i]['name']}")


EXP_GROUPS = [
    {
        'name': '2024.04.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_AssocFullsymmCommu',
        'sub_exp': 2,
        'check_point': 'checkpoint_10000.pt',
    }, {
        'name': '2024.04.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_AssocFullsymmCommu',
        'sub_exp': 1,
        'check_point': 'checkpoint_10000.pt',
    }, {
        'name': '2024.04.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_Nothing',
        'sub_exp': 6,
        'check_point': 'checkpoint_9500.pt',
    }, {
        'name': '2024.04.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_Nothing',
        'sub_exp': 1,
        'check_point': 'checkpoint_10000.pt',
    },
]
if __name__ == "__main__":
    for exp_group in EXP_GROUPS:
        eval_an_sub_exp(exp_group['name'], exp_group['sub_exp'], exp_group['check_point'], eval_config=EVAL_CONFIG, result_root=RESULT_ROOT)



