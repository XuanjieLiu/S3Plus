import os
import shutil
import argparse
import random
from glob import glob
from tqdm import tqdm
from copy import deepcopy


def split_train_val(data_dir, output_dir, val_percentage=0.1):
    """
    data_dir: directory directly containing wav files
    output_dir: directory to store the split data
    val_percentage: percentage of data to be used as validation set
    """
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    npy_paths = glob(os.path.join(data_dir, "*.npy"))
    n_val = int(len(npy_paths) * val_percentage)
    print(f"{len(npy_paths)} wav files found.")
    random.shuffle(npy_paths)
    for i, npy_path in tqdm(enumerate(npy_paths)):
        if i < n_val:
            shutil.copy(npy_path, val_dir)
        elif i >= n_val:
            shutil.copy(npy_path, train_dir)

    print(
        f"Splited into {len(glob(os.path.join(train_dir, '*.npy')))} training files and {len(glob(os.path.join(val_dir, '*.npy')))} validation files."
    )


def split_train_val_nonrandom(data_dir, output_dir, val_percentage=0.1, n_total=20):
    """
    Function for datasets with the need of non-random splitting.
    Each file should be named as "xxx_number.npy". The number is used to split the data.

    data_dir: directory directly containing wav files
    output_dir: directory to store the split data
    val_percentage: percentage of data to be used as validation set
    n_total: total number of indices.
    """
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    npy_paths = glob(os.path.join(data_dir, "*.npy"))
    for npy_path in tqdm(npy_paths):
        npy_name = os.path.basename(npy_path)
        if int(npy_name.split("_")[1].split(".")[0]) < n_total * val_percentage:
            shutil.copy(npy_path, val_dir)
        else:
            shutil.copy(npy_path, train_dir)

    print(
        f"Splited into {len(glob(os.path.join(train_dir, '*.npy')))} training files and {len(glob(os.path.join(val_dir, '*.npy')))} validation files."
    )


def split_ood_few_shot(data_dir, output_dir, n_shots=[1, 5, 10]):
    """
    data_dir: directory directly containing .png files
    shots: number of shots for each class. The class label (style label) is the last word of the file name.
    files selected for few-shot learning will not be shown in the test set.
    """
    png_paths = glob(os.path.join(data_dir, "*.npy"))
    test_png_paths = deepcopy(png_paths)
    for n_shot in n_shots:
        shots = {}
        shot_dir = os.path.join(output_dir, f"{n_shot}_shot")
        os.makedirs(shot_dir, exist_ok=True)
        for png_path in png_paths:
            png_name = os.path.basename(png_path)
            s_label = png_name.split("_")[0]
            if s_label not in shots:
                shots[s_label] = []
            if len(shots[s_label]) < n_shot:
                shots[s_label].append(png_path)
                if png_path in test_png_paths:
                    test_png_paths.remove(png_path)
            else:
                continue
        for s_label, paths in shots.items():
            for path in paths:
                shutil.copy(path, shot_dir)
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(test_dir, exist_ok=True)
    for png_path in test_png_paths:
        shutil.copy(png_path, test_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="../data/Wikifonia_melody_magspec"
    )
    parser.add_argument(
        "--output_dir", type=str, default="../data/WikifoniaMelodyMagspec"
    )
    parser.add_argument("--val_percentage", type=float, default=0.1)
    parser.add_argument("--random", type=int, default=1)
    parser.add_argument("--n_total", type=int, default=20)
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    val_percentage = args.val_percentage
    if args.random > 0:
        split_train_val(data_dir, output_dir, val_percentage)
    else:
        split_train_val_nonrandom(data_dir, output_dir, val_percentage, args.n_total)
