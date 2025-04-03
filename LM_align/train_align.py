import os
import torch.optim as optim
import torch.nn as nn
from vis_funcs import visualize_alignment
from loader_VQSPS import VQSPSLoader
from VQ.VQVAE import MultiVectorQuantizer
from loader_dino import load_dino_vit_s8
from utils import VQSPS_EXP_ROOT
from torch.utils.data import Dataset, DataLoader
from VQ.common_func import load_config_from_exp_name
from projector import FCProjector
from FruitRecognitionDataset import FruitRecognitionDataset, random_mask_batch
from synthData.SynthImgsDataset import PreGeneratedDataset, onlineGenDataset, OBJ_LIST_2
from shared import *
import wandb
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from LM_align.VisionModels import LightweightViT, SimpleCNN


WANDB_API_KEY="532007cd7a07c1aa0d1194049c3231dadd1d418e"
wandb.login(key=WANDB_API_KEY)
PROJECT_DIR = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../')
RESULTS_DIR = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/results/')
print(f'Results path: {RESULTS_DIR}')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_TYPE_FRUIT = 'fruit'
DATA_TYPE_SYNTH = 'synth'
DATA_TYPE_SYNTH_ONLINE = 'synth_online'
STAGE_TRAIN = 'train'
STAGE_VAL = 'val'
STAGE_VAL_OOD = 'val_ood'
VISION_MODEL_DINO = 'dino'
VISION_MODEL_LIGHTViT = 'lightViT'
VISION_MODEL_CNN = 'cnn'

CONFIG = {
    'name': '25.03.24.align_Synth_online_dino_adjustCollapse_noZero_3',
    'dataset_anchor': os.path.join(PROJECT_DIR, 'fruit_recognition_dataset_oneFruit'),
    'subset_list': ['Apple/Apple A', 'Apple/Apple D'],
    'VQSPS': {
        'EXP_NAME': '2024.02.03_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_AssocSymmCommuAll',
        'CHECK_POINT_NAME': '2/checkpoint_50000.pt',
    },
    # Dino Projector
    'PROJECTOR': {
        'HIDDEN_PARAM': [1024, 1024, 1024],
        'IS_BATCH_NORM': False,
        'IS_LAYER_NORM': False,
        'NG_SLOPE': 0.01,
    },

    # LightVisionModel Projector
    # 'PROJECTOR': {
    #     'HIDDEN_PARAM': [256, 256],
    #     'IS_BATCH_NORM': False,
    #     'IS_LAYER_NORM': False,
    #     'NG_SLOPE': 0.01,
    # },
    'project_name': 'LM_align_2.20',
    'vision_model': VISION_MODEL_DINO,
    'vision_out_dim': 256,
    'ALIGN': {
        'LR': 0.0001,
        'BATCH_SIZE': 32,
        'EPOCHS': 100,
        'commitment_scalar': 1,
        'embedding_scalar': 0.0,
        'collapse_scalar': 1,
        'collapse_multiplier': 1, # before was 2.0
        'is_use_anchor': False,
        'plus_scalar': 1, # before was 100
        # 'CODES': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'CODES': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        # 'CODES': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20],
    },
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
        'train_num_samples': 1024,
        'train_reuse_times': 4, # 4 for dino
    },
    
}

def calc_accuracy(pred_label, true_label):
    return np.sum(pred_label == true_label) / len(true_label)

def collapse_loss_func(margin, collapse_multiplier=1):
    def collapse_loss(a, b, c):
        # 对每个样本计算pairwise欧氏距离
        d_ab = torch.norm(a - b, p=2, dim=1)
        d_ac = torch.norm(a - c, p=2, dim=1)
        d_bc = torch.norm(b - c, p=2, dim=1)
        
        # 对每个样本取三个pairwise距离的最大值
        d_max = torch.max(torch.stack([d_ab, d_ac, d_bc], dim=1), dim=1)[0] * collapse_multiplier
        
        # 当最大距离小于margin时，对该样本施加惩罚
        loss = F.relu(margin - d_max)
        
        # 返回整个batch的平均loss
        return loss.mean()
    return collapse_loss

def load_VQSPS_loader(config=CONFIG):
    vqsps_exp_name = config['VQSPS']['EXP_NAME']
    vqsps_config = load_config_from_exp_name(vqsps_exp_name)
    model_path = os.path.join(VQSPS_EXP_ROOT, vqsps_exp_name, config['VQSPS']['CHECK_POINT_NAME'])
    return VQSPSLoader(vqsps_config, model_path)


def load_fc_projector(input_dim, output_dim, config=CONFIG):
    return FCProjector(
        input_dim, output_dim,
        hidden_param=config['PROJECTOR']['HIDDEN_PARAM'],
        is_batch_norm=config['PROJECTOR']['IS_BATCH_NORM'],
        is_layer_norm=config['PROJECTOR']['IS_LAYER_NORM']
    )


def load_dataloader_anchor(config, data_type=CONFIG['data_type']):
    if data_type == DATA_TYPE_FRUIT:
        dataset = FruitRecognitionDataset(config['data_fruit']['dataset_anchor'], config['data_fruit']['subset_list'])
        return DataLoader(dataset, batch_size=config['ALIGN']['BATCH_SIZE'], shuffle=True)
    if data_type == DATA_TYPE_SYNTH:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        dataset = PreGeneratedDataset(config['data_synth']['dataset_anchor'], transform=transform)
        return DataLoader(dataset, batch_size=config['ALIGN']['BATCH_SIZE'], shuffle=True)

def load_dataloader(config, data_type=CONFIG['data_type'], type='train'):
    if data_type == DATA_TYPE_FRUIT:
        dataset = FruitRecognitionDataset()
        return DataLoader(dataset, batch_size=config['ALIGN']['BATCH_SIZE'], shuffle=True)
    if data_type == DATA_TYPE_SYNTH:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        if type == 'train':
            dataset = PreGeneratedDataset(config['data_synth']['train_dir'], transform=transform)
            return DataLoader(dataset, batch_size=config['ALIGN']['BATCH_SIZE'], shuffle=True)
        if type == 'val':
            dataset = PreGeneratedDataset(config['data_synth']['val_dir'], transform=transform)
            return DataLoader(dataset, batch_size=256, shuffle=True)
    if data_type == DATA_TYPE_SYNTH_ONLINE:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        num_samples = config['data_synth_online']['train_num_samples']
        reuse_times = config['data_synth_online']['train_reuse_times']
        if type == 'train':
            dataset = onlineGenDataset(num_samples=num_samples, reuse_times=reuse_times, transform=transform)
            return DataLoader(dataset, batch_size=config['ALIGN']['BATCH_SIZE'], shuffle=False)
        if type == 'val':
            dataset = onlineGenDataset(num_samples=256, reuse_times=1, transform=transform)
            return DataLoader(dataset, batch_size=config['ALIGN']['BATCH_SIZE'], shuffle=False)
        if type == 'val_ood':
            dataset = onlineGenDataset(num_samples=256, reuse_times=1, transform=transform, obj_list=OBJ_LIST_2)
            return DataLoader(dataset, batch_size=config['ALIGN']['BATCH_SIZE'], shuffle=False)
        

def load_vision_model(config=CONFIG):
    vision_out_dim = config['vision_out_dim']
    if config['vision_model'] == 'dino':
        return load_dino_vit_s8().to(DEVICE), 384
    if config['vision_model'] == VISION_MODEL_LIGHTViT:
        return LightweightViT(out_dim=vision_out_dim).to(DEVICE), vision_out_dim
    if config['vision_model'] == VISION_MODEL_CNN:
        return SimpleCNN(output_dim=vision_out_dim).to(DEVICE), vision_out_dim
        

class AlignTrain:
    def __init__(self, config):
        self.config = config
        self.data_type = config['data_type']
        self.data_loader_train = load_dataloader(config, type='train')
        self.data_loader_val = load_dataloader(config, type='val')
        self.data_loader_val_ood = load_dataloader(config, type='val_ood')
        self.data_loader_anchor = load_dataloader_anchor(config)
        self.model_vision, proj_in_dim = load_vision_model(config)
        self.sps_loader = load_VQSPS_loader(config)
        self.model_sps = self.sps_loader.model.to(DEVICE)
        proj_out_dim = self.sps_loader.latent_code_1
        self.model_proj = load_fc_projector(proj_in_dim, proj_out_dim, config).to(DEVICE)
        self.results_dir = os.path.join(RESULTS_DIR, config['name'])
        self.batch_size = config['ALIGN']['BATCH_SIZE']
        self.anchor_1_z = self.sps_loader.num_z_c[self.sps_loader.num_labels.index(1)]
        self.anchor_1_z_batch = torch.tensor(self.anchor_1_z).unsqueeze(0).repeat(self.batch_size, 1).to(DEVICE)
        self.codes = self.config['ALIGN']['CODES']
        self.vq_layer, self.min_margin = self.init_codebook()
        self.criterion = nn.MSELoss()
        self.train_step = 0
        self.collapse_loss = collapse_loss_func(self.min_margin, config['ALIGN']['collapse_multiplier'])
        self.collapse_scalar = config['ALIGN']['collapse_scalar']
        

    def init_codebook(self):
        codes_index = [self.sps_loader.num_labels.index(code) for code in self.codes]
        codebook_emb = self.sps_loader.num_z_c[codes_index]
        num_embeddings = len(self.codes)
        min_margin = torch.min(torch.pdist(torch.tensor(codebook_emb), p=2))
        vq_layer = MultiVectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=codebook_emb.shape[1],
            commitment_cost=self.config['ALIGN']['commitment_scalar'],
            embedding_cost=self.config['ALIGN']['embedding_scalar'],
            init_embs=codebook_emb
        ).to(DEVICE)
        return vq_layer, min_margin

    def embs2label(self, embs):
        embs_idx = self.vq_layer.get_code_indices(embs).cpu().squeeze(1).numpy()
        embs_label = np.array(self.codes)[embs_idx]
        return embs_label


    def databatch2imgs(self, data_batch):
        if self.data_type == DATA_TYPE_FRUIT:
            img, _ = data_batch
            img = img.to(DEVICE)
            masked_a, masked_b = random_mask_batch(img)
            return masked_a, masked_b, img
        if self.data_type == DATA_TYPE_SYNTH or self.data_type == DATA_TYPE_SYNTH_ONLINE:
            img_a = data_batch['image_a'].to(DEVICE)
            img_b = data_batch['image_b'].to(DEVICE)
            img_c = data_batch['image_c'].to(DEVICE)
            label_a = data_batch['label']['a'].numpy()
            label_b = data_batch['label']['b'].numpy()
            label_c = data_batch['label']['c'].numpy()
            return img_a, img_b, img_c, label_a, label_b, label_c
        
    def databatch2imgs_anchor(self, data_batch):
        if self.data_type == DATA_TYPE_FRUIT:
            img_anchor = data_batch[0].to(DEVICE)
            return img_anchor
        if self.data_type == DATA_TYPE_SYNTH:
            img_c = data_batch['image_c'].to(DEVICE)
            return img_c

    def init_wandb(self):
        wandb.init(
            project=self.config['project_name'], 
            name=self.config['name'],
            config=self.config,
        )
        
    def img2emb(self, img):
        with torch.no_grad():
            vision_out = self.model_vision(img)
        proj_out = self.model_proj(vision_out)
        # e, e_q_loss = self.model_sps.vq_layer(proj_out)
        e, e_q_loss = self.vq_layer(proj_out)
        return e, e_q_loss, proj_out
    
    def plus(self, z_a, z_b):
        comb = torch.cat((z_a, z_b), dim=-1)
        z_plus = self.model_sps.plus_net(comb)
        e_plus, e_q_loss = self.vq_layer(z_plus)
        return e_plus, e_q_loss, z_plus
    
    def one_epoch(self, epoch, data_loader, optimizer=None, stage=STAGE_TRAIN):
        for i, data_batch in enumerate(data_loader):
            print(f'Stage: {stage}, Epoch: {epoch}, Batch: {i}, Train step: {self.train_step}')
            if optimizer is not None:
                optimizer.zero_grad()
            img_a, img_b, img_c, label_a, label_b, label_c = self.databatch2imgs(data_batch)
            e_a, e_q_loss_a, z_a = self.img2emb(img_a)
            e_b, e_q_loss_b, z_b = self.img2emb(img_b)
            e_c, e_q_loss_c, z_c = self.img2emb(img_c)
            e_ab, e_q_loss_ab, z_ab = self.plus(e_a, e_b)
            plus_loss = self.criterion(e_ab.detach(), e_c) + self.criterion(e_ab, e_c.detach())
            plus_loss = plus_loss * self.config['ALIGN']['plus_scalar']

            collapse_loss_c = self.collapse_loss(e_a, e_b, e_c)
            collapse_loss_ab = self.collapse_loss(e_a, e_b, e_ab)
            collapse_loss_all = collapse_loss_c + collapse_loss_ab
            collapse_loss = collapse_loss_all * self.collapse_scalar

            # Calculate accuracy
            e_all = torch.cat((e_a, e_b, e_c), dim=0)
            label_all = np.concatenate((label_a, label_b, label_c))
            pred_label = self.embs2label(e_all)
            accuracy = calc_accuracy(pred_label, label_all)

            # Anchor loss
            anchor_loss = torch.zeros(1).to(DEVICE)
            if self.config['ALIGN']['is_use_anchor']:
                for j, anchor_batch in enumerate(self.data_loader_anchor):
                    img_anchor = self.databatch2imgs_anchor(anchor_batch)
                    e_anchor, e_q_loss_anchor, z_anchor = self.img2emb(img_anchor)
                    anchor_loss = self.criterion(e_anchor, self.anchor_1_z_batch)
                    break
            
            all_loss = e_q_loss_a + e_q_loss_b + e_q_loss_c + e_q_loss_ab + plus_loss + anchor_loss + collapse_loss
            if optimizer is not None:
                all_loss.backward()
                optimizer.step()

            loss_dict = {
                f'{stage}_e_q_loss_a': e_q_loss_a,
                f'{stage}_e_q_loss_b': e_q_loss_b,
                f'{stage}_e_q_loss_c': e_q_loss_c,
                f'{stage}_e_q_loss_ab': e_q_loss_ab,
                f'{stage}_plus_loss': plus_loss,
                f'{stage}_anchor_loss': anchor_loss,
                f'{stage}_collapse_loss': collapse_loss,
                f'{stage}_collapse_loss_mean': collapse_loss_all,
                f'{stage}_all_loss': all_loss,
                f'{stage}_train_step': self.train_step,
                f'{stage}_accuracy': accuracy,
            }

            if stage == STAGE_TRAIN:
                self.train_step += 1

            wandb.log(loss_dict)
            if self.train_step % 50 == 0 or stage == STAGE_VAL or stage == STAGE_VAL_OOD: # This command for dino
            # if stage == STAGE_VAL or stage == STAGE_VAL_OOD:
                visualize_alignment(
                    self.sps_loader.num_z_c, self.sps_loader.num_labels,
                    [img_a[0].cpu().detach().numpy(), img_b[0].cpu().detach().numpy(), img_c[0].cpu().detach().numpy()],
                    [e_a[0].cpu().detach().numpy(), e_b[0].cpu().detach().numpy(), e_c[0].cpu().detach().numpy()],
                    [z_a[0].cpu().detach().numpy(), z_b[0].cpu().detach().numpy(), z_c[0].cpu().detach().numpy()],
                    e_ab[0].cpu().detach().numpy(),
                    z_ab[0].cpu().detach().numpy(),
                    save_path=os.path.join(self.results_dir ,f'{stage}_step_{self.train_step}.png')
                )
            if stage == STAGE_VAL or stage == STAGE_VAL_OOD:
                break

    def train(self):
        self.init_wandb()
        os.makedirs(self.results_dir, exist_ok=True)
        if self.config['vision_model'] == VISION_MODEL_DINO:
            optimizer = optim.Adam(self.model_proj.parameters(), lr=self.config['ALIGN']['LR'])
        elif self.config['vision_model'] == VISION_MODEL_LIGHTViT or self.config['vision_model'] == VISION_MODEL_CNN:
            optimizer = optim.Adam(list(self.model_proj.parameters()) + list(self.model_vision.parameters()), lr=self.config['ALIGN']['LR'])
        for epoch in range(self.config['ALIGN']['EPOCHS']):
            self.one_epoch(epoch, self.data_loader_train, optimizer, stage=STAGE_TRAIN)
            with torch.no_grad():
                self.one_epoch(epoch, self.data_loader_val, stage=STAGE_VAL)
                self.one_epoch(epoch, self.data_loader_val_ood, stage=STAGE_VAL_OOD)
        self.train_step = 0


if __name__ == "__main__":
    align_train = AlignTrain(CONFIG)
    align_train.train()

                




