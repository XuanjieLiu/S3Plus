import os
from VQVAE import VQVAE
from shared import *
from dataloader_plus import Dataset
from torch.utils.data import DataLoader
from common_func import parse_label, load_config_from_exp_name, EXP_ROOT
from torchvision.utils import save_image


class OperResult:
    def __init__(self, label_a, label_b, label_c, plus_c_z, recon, style=''):
        self.plus_c_z = plus_c_z
        self.label_a = label_a
        self.label_b = label_b
        self.label_c = label_c
        self.recon = recon
        self.style = style

    def save_recon(self, path):
        prefix = f'{self.style}.' if self.style != '' else ''
        name = f'{prefix}{self.label_c}.{self.label_a}_{self.label_b}.png'
        save_image(self.recon, os.path.join(path, name))


class CommonEval:
    def __init__(self, config, model_path=None, loaded_model: VQVAE = None):
        self.config = config
        self.zc_dim = config['latent_embedding_1'] * config['embedding_dim']
        if loaded_model is not None:
            self.model = loaded_model
        elif model_path is not None:
            self.model = VQVAE(config).to(DEVICE)
            self.plus_model_reload(model_path)
            print(f"Model is loaded from {model_path}")
        else:
            self.model = VQVAE(config).to(DEVICE)
            print("No model is loaded")
        self.model.eval()

    def plus_model_reload(self, checkpoint_path):
        self.model.load_state_dict(self.model.load_tensor(checkpoint_path))
        print('Model loaded from {}'.format(checkpoint_path))

    def eval_oper(self, data_path, result_path):
        os.makedirs(result_path, exist_ok=True)
        dataset = Dataset(data_path)
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        styles = [('-').join(f.split('-')[2:]) for f in dataset.f_list]
        all_oper_result = []
        for batch_ndx, sample in enumerate(loader):
            data, labels = sample
            label_a = [parse_label(x) for x in labels[0]]
            label_b = [parse_label(x) for x in labels[1]]
            label_c = [parse_label(x) for x in labels[2]]
            plus_result = []
            recon_a, recon_b, recon_c, recon_ab, e_a, e_b, e_c, e_ab = self.model.forward(data)
            for i in range(0, e_a.size(0)):
                plus_result.append(OperResult(label_a[i], label_b[i], label_c[i], e_ab[i], recon_ab[i]))
            all_oper_result.extend(plus_result)

        for i in range(len(all_oper_result)):
            all_oper_result[i].style = styles[i]
            all_oper_result[i].save_recon(result_path)


if __name__ == "__main__":
    EXP_NAME = '2023.12.17_multiStyle_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_2_realPair_noAssoc'
    SUB_EXP = 12
    CHECK_POINT = 'checkpoint_10000.pt'
    check_point_path = os.path.join(EXP_ROOT, EXP_NAME, str(SUB_EXP), CHECK_POINT)
    PLUS_RECON_RESULT_PATH = os.path.join(EXP_ROOT, EXP_NAME, str(SUB_EXP), 'plus_recon_result')
    exp_path = os.path.join(EXP_ROOT, EXP_NAME)
    config = load_config_from_exp_name(EXP_NAME)
    evaler = CommonEval(config, check_point_path)

    # Eval on train data
    train_data_path = config['train_data_path']
    train_result_path = os.path.join(EXP_ROOT, EXP_NAME, str(SUB_EXP), 'plus_train_recon_result')
    evaler.eval_oper(train_data_path, train_result_path)

    # Eval on test_1 data
    test_1_data_path = config['plus_eval_set_path']
    test_1_result_path = os.path.join(EXP_ROOT, EXP_NAME, str(SUB_EXP), 'plus_test_1_recon_result')
    evaler.eval_oper(test_1_data_path, test_1_result_path)

    # Eval on test_2 data
    test_2_data_path = config['plus_eval_set_path_2']
    if test_2_data_path is None:
        print('No test_2 data path is found')
    else:
        test_2_result_path = os.path.join(EXP_ROOT, EXP_NAME, str(SUB_EXP), 'plus_test_2_recon_result')
        evaler.eval_oper(test_2_data_path, test_2_result_path)
