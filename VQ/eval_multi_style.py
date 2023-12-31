from dataloader import SingleImgDataset, load_enc_eval_data
from typing import List
from VQ.VQVAE import VQVAE
from shared import *
from torch.utils.data import DataLoader
import sys
import os
from importlib import reload


def load_idx_and_label(loader: DataLoader, loaded_model: VQVAE):
    def encode_func(x):
        z = loaded_model.batch_encode_to_z(x)[0]
        idx = loaded_model.find_indices(z, True, False)
        return idx
    idx, num_labels = load_enc_eval_data(
        loader,
        encode_func
    )
    return [int(n[0]) for n in idx.tolist()], num_labels




def eval_accu(base_idxs: List[int], base_labels: List[int], eval_idxs: List[int], eval_labels: List[int]):
    label_set = list(set(base_labels))
    base_dict = {}
    for label in label_set:
        base_dict[str(label)] = []
    for i in range(len(base_idxs)):
        base_dict[str(base_labels[i])].append(base_idxs[i])
    mode_dict = {}
    for key, value in base_dict.items():
        mode = find_mode(value)
        assert mode is not None
        mode_dict[key] = mode
    correct = 0
    for i in range(len(eval_idxs)):
        if mode_dict[str(eval_labels[i])] == eval_idxs[i]:
            correct += 1
    return correct / len(eval_idxs)




def find_mode(int_list: List[int]):
    int_set = list(set(int_list))
    int_dict = {}
    for i in int_set:
        int_dict[str(i)] = 0
    for i in int_list:
        int_dict[str(i)] += 1
    max_count = 0
    mode = None
    for i in int_dict.keys():
        if int_dict[i] > max_count:
            max_count = int_dict[i]
            mode = int(i)
    return mode


class AccuEval:
    def __init__(self, config, train_set_path, eval_set_path, model: VQVAE = None):
        self.config = config
        self.model = VQVAE(config).to(DEVICE) if model is None else model
        self.model.eval()
        self.embedding_dim = config['embedding_dim']
        self.multi_num_embeddings = config['multi_num_embeddings']
        if self.multi_num_embeddings is None:
            self.latent_embedding_1 = config['latent_embedding_1']
        else:
            self.latent_embedding_1 = len(self.multi_num_embeddings)
        self.latent_code_1 = self.latent_embedding_1 * self.embedding_dim
        self.batch_size = config['batch_size']
        train_set = SingleImgDataset(train_set_path)
        eval_set = SingleImgDataset(eval_set_path)
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size)
        self.eval_loader = DataLoader(eval_set, batch_size=self.batch_size)

    def eval_check_point(self, checkpoint_path):
        self.model_reload(checkpoint_path)
        train_idx, train_labels = load_idx_and_label(self.train_loader, self.model)
        eval_idx, eval_labels = load_idx_and_label(self.eval_loader, self.model)
        accu = eval_accu(train_idx, train_labels, eval_idx, eval_labels)
        return accu

    def model_reload(self, checkpoint_path):
        self.model.load_state_dict(self.model.load_tensor(checkpoint_path))
        print('Model loaded from {}'.format(checkpoint_path))




if __name__ == "__main__":
    DATASET_ROOT = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../dataset/')
    EXP_ROOT_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/exp')
    EXP_NAME = '2023.12.17_multiStyle_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_2_realPair'
    SUB_EXP = 1
    CHECK_POINT = 'curr_model.pt'
    exp_path = os.path.join(EXP_ROOT_PATH, EXP_NAME)
    check_point_path = os.path.join(exp_path, str(SUB_EXP), CHECK_POINT)
    sys.path.append(exp_path)
    os.chdir(exp_path)
    print(f'Exp path: {exp_path}')
    t_config = __import__('train_config')
    reload(t_config)
    train_set_path = os.path.join(DATASET_ROOT, 'multi_style_eval_(0,20)_FixedPos_TrainStyle')
    # eval_set_path = os.path.join(DATASET_ROOT, 'multi_style_eval_(0,20)_FixedPos_TrainStyle')
    eval_set_path = os.path.join(DATASET_ROOT, 'multi_style_eval_(0,20)_FixedPos_newShapeColor')
    evaler = AccuEval(t_config.CONFIG, train_set_path, eval_set_path)
    accu = evaler.eval_check_point(check_point_path)
    print(f'Accu: {accu}')