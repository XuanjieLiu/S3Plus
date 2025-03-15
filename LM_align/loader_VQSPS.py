from VQ.VQVAE import VQVAE
from VQ.common_func import load_config_from_exp_name
from VQ.eval_common import CommonEvaler
from dataloader import SingleImgDataset, load_enc_eval_data
from torch.utils.data import DataLoader
from utils import VQSPS_EXP_ROOT
import os


EXP_NAME = '2024.04.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_multiStyle_AssocFullsymmCommu'
EXP_PATH = os.path.join(VQSPS_EXP_ROOT, EXP_NAME)
checkpoint_name = f'1/checkpoint_10000.pt'
MODEL_PATH = os.path.join(EXP_PATH, checkpoint_name)


# Load the VQ-VAE model
config = load_config_from_exp_name(EXP_NAME)



class VQSPSLoader(CommonEvaler):
    def __init__(self, config, model_path):
        super().__init__(config, model_path)
        dataset_path = config['single_img_eval_set_path']
        single_img_eval_set = SingleImgDataset(dataset_path)
        self.single_img_eval_loader = DataLoader(single_img_eval_set, batch_size=256)
        self.num_z_c, self.num_labels = self.get_z_labels()


    def get_z_labels(self):
        num_z, num_labels = load_enc_eval_data(
            self.single_img_eval_loader,
            lambda x:
                self.model.batch_encode_to_z(x)[0]
        )
        num_z = num_z.cpu().detach().numpy()
        num_z_c = num_z[:, :self.latent_code_1]
        return num_z_c, num_labels

