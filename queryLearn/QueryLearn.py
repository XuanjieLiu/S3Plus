import torch.nn as nn
import os
from VQ.VQVAE import VQVAE
from VQ.common_func import load_config_from_exp_name
from VQ.eval_common import CommonEvaler
from shared import DEVICE


VQSPS_EXP_ROOT = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../VQ/exp/')


def init_query_learner(query_learner_config):
    in_dim = query_learner_config['in_dim']
    out_dim = query_learner_config['out_dim']
    return nn.Linear(in_dim, out_dim)

def load_VQSPS_loader(config):
    vqsps_exp_name = config['VQSPS']['EXP_NAME']
    vqsps_config = load_config_from_exp_name(vqsps_exp_name)
    model_path = os.path.join(VQSPS_EXP_ROOT, vqsps_exp_name, config['VQSPS']['CHECK_POINT_NAME'])
    return CommonEvaler(vqsps_config, model_path), vqsps_config




class QueryLearn:
    def __init__(self, config, model_path=None, loaded_model: VQVAE = None):
        sps_model, sps_config = load_VQSPS_loader(config)
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
        self.query_learner = init_query_learner(config['query_learner']).to(DEVICE)


    def plus_model_reload(self, checkpoint_path):
        self.model.load_state_dict(self.model.load_tensor(checkpoint_path))
        print('Model loaded from {}'.format(checkpoint_path))