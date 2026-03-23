from typing import List, Callable, Any
import torch
from torch.utils.data import ConcatDataset, DataLoader
from dataloader import SingleImgDataset
from dataloader_plus import MultiImgDataset, make_dataset_trans



def make_dataset(data_path: str | List[str], dataset_cls: Callable[..., Any],
                     transform: Any = None, augment_times: int = 1) -> ConcatDataset:
    print(f"data_path is: {data_path}")
    path_list = data_path if isinstance(data_path, list) else [data_path]
    print(f"Making dataset from paths: {path_list}")
    # 如果有多个 set_path, 创建多个 dataset 并合并
    datasets = ConcatDataset([dataset_cls(path, augment_times=augment_times, transform=transform)
                              for path in path_list])
    return datasets


def init_dataloaders(config):
    aug_t = config.get('augment_times', 1)
    blur_cfg = config.get('blur_config', None)
    is_blur = config.get('is_blur', False)
    trans = make_dataset_trans(is_blur, blur_cfg) if is_blur else None
    n_workers = config.get('num_workers', 0)
    loader_config = {
        'batch_size': config['batch_size'],
        'shuffle': True,
        'num_workers': n_workers,
        'persistent_workers': True if n_workers > 0 else False,
    }
    # 新增：从 config 读取可选的随机种子
    split_seed = config.get('random_split_seed', None)
    split_generator = None
    if split_seed is not None:
        # 只用于 random_split，本身不影响全局 RNG
        split_generator = torch.Generator()
        # 建议显式转 int，避免从 JSON 读取成 str 报错
        split_generator.manual_seed(int(split_seed))
    if config.get('is_random_split_data', False):
        print("Using random split data")
        train_ratio = config['train_data_ratio']
        dataset_all = make_dataset(config['train_data_path'], MultiImgDataset, transform=trans, augment_times=aug_t)
        train_size = int(len(dataset_all) * train_ratio)
        eval_size = len(dataset_all) - train_size
        # 仅当提供了 split_seed 时，才把 generator 传进去；否则保持原行为
        if split_generator is None:
            train_dataset, eval_dataset = torch.utils.data.random_split(dataset_all, [train_size, eval_size])
        else:
            train_dataset, eval_dataset = torch.utils.data.random_split(
                dataset_all, [train_size, eval_size], generator=split_generator
            )
        plus_train_loader = DataLoader(train_dataset, **loader_config)
        plus_eval_loader = DataLoader(eval_dataset, **loader_config)
    else:
        print("Using predefined datasets")
        plus_train_set = make_dataset(config['train_data_path'], MultiImgDataset, transform=trans, augment_times=aug_t)
        plus_eval_set = make_dataset(config['plus_eval_set_path'], MultiImgDataset, transform=trans, augment_times=aug_t)
        plus_train_loader = DataLoader(plus_train_set, **loader_config)
        plus_eval_loader = DataLoader(plus_eval_set, **loader_config)
    if config.get('single_img_eval_set_path', None) is not None:
        single_img_eval_set = make_dataset(config['single_img_eval_set_path'], SingleImgDataset, transform=trans, augment_times=aug_t)
        single_img_eval_loader = DataLoader(single_img_eval_set, **loader_config)
    else:
        single_img_eval_loader = None
    return plus_train_loader, plus_eval_loader, single_img_eval_loader