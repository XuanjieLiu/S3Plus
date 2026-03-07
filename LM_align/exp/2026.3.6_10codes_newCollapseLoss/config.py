import os
from LM_align.common_config import (VISION_MODEL_DINO, DATA_TYPE_SYNTH_ONLINE, PROJECT_DIR)
# 获取当前文件所在的上一级文件夹名称
parent_folder_name = os.path.basename(os.path.dirname(__file__))
CONFIG = {
    'name': parent_folder_name,
    'subset_list': ['Apple/Apple A', 'Apple/Apple D'],
    'VQSPS': {
        'EXP_NAME': '2025.05.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Fullsymm',
        'CHECK_POINT_NAME': '1/checkpoint_48000.pt',
    },
    # Dino Projector
    'PROJECTOR': {
        'HIDDEN_PARAM': [1024, 1024, 1024],
        'IS_BATCH_NORM': False,
        'IS_LAYER_NORM': False,
        'NG_SLOPE': 0.01,
    },
    'project_name': 'LM_align_2.20',
    'vision_model': VISION_MODEL_DINO,
    'group': "experiment_replicate",
    'vision_out_dim': 256,
    'ALIGN': {
        'LR': 0.0001,
        'BATCH_SIZE': 32,
        'EPOCHS': 30,
        'commitment_scalar': 1,
        'embedding_scalar': 0.0,
        'collapse_scalar': 1,
        'collapse_multiplier': 1,  # before was 2.0
        'is_use_anchor': False,
        'plus_scalar': 1,  # before was 100
        # 'CODES': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'CODES': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        # 'CODES': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20],
        'ckpt_interval': 1,
        'log_interval': 1,
        'result_dir': 'Results/',
        'ckpt_name': 'curr_model.pt',
        'train_record_path': "Train_record.txt",
        'val_record_path': "Val_record.txt",
        'val_ood_record_path': "Val_ood_record.txt",
    },
    'num_sub_exp': 20,
    'data_type': DATA_TYPE_SYNTH_ONLINE,
    'data_fruit': {
        'dataset_anchor': os.path.join(PROJECT_DIR, 'fruit_recognition_dataset_oneFruit'),
        'subset_list': ['Apple/Apple A', 'Apple/Apple D'],
    },
    'data_synth': {
        'train_dir': os.path.join(PROJECT_DIR, 'LM_align/synthData/pre_generated_data'),
        'val_dir': os.path.join(PROJECT_DIR, 'LM_align/synthData/pre_generated_data_sharedBoxes_val'),
        'dataset_anchor': os.path.join(PROJECT_DIR, 'LM_align/synthData/pre_generated_data_anchor_1'),
    },
    'data_synth_online': {
        'train_num_samples': 1024,  # 1024 for dino
        'train_reuse_times': 4,  # 4 for dino
        # 'train_reuse_times': 16, # 16 for vit and cnn
    },
}

