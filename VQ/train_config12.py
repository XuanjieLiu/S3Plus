import math
import os

data_root = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../dataset')
CONFIG = {
    'train_data_path': f"{data_root}/PlusPair-(1,7)-FixedPos/train",
    'eval_path_1': f"{data_root}/(1,16)-FixedPos-4Color",
    'latent_embedding_1': 2,
    'latent_embedding_2': 2,
    'latent_code_2': 2,
    'embedding_dim': 8,
    'embeddings_num': 8,
    'kld_loss_scalar': 0.0,
    'checkpoint_interval': 1000,
    'learning_rate': 1e-4,
    'scheduler_base_num': 0.99999,
    'max_iter_num': 6001,
    'model_path': 'curr_model.pt',
    'train_result_path': 'TrainingResults/',
    'train_record_path': "Train_record.txt",
    'eval_record_path': "Eval_record.txt",
    'log_interval': 10,
    'eval_interval': 1000,
    'is_save_img': True,
    'batch_size': 32,
    'z_plus_loss_scalar': 0.5,
    'commutative_z_loss_scalar': 0.5,
    'associative_z_loss_scalar': 0.5,
    'plus_recon_loss_scalar': 1.5,
    'min_loss_scalar': 0.00001,
    'K': 1024,
    'assoc_aug_range': (-3, 3),
    'commitment_scalar': 0.025,
    'embedding_scalar': 0.01,
    'isVQStyle': False,
    'plus_by_embedding': False,
    'VQPlus_eqLoss_scalar': 0.5,
    'network_config': {
        'enc_dec': {
            'img_channel': 3,
            'last_H': 4,
            'last_W': 4,
            'first_ch_num': 64,
        },
        'plus': {
            'plus_unit': 4,
        }
    },
}
