"""
This is the dataloader module for the music melody spectrogram dataset.
This dataset should have clear labels of pitches and timbres.
"""

import os
import random
import argparse
import itertools
from glob import glob

import numpy as np
import librosa as lr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
from torchaudio.transforms import Spectrogram, MelScale

from dataset.music_melody.gen_melody_from_samples import PseudoMelGen
from utils.training_utils import setup_seed

S_LIST = [
    "Soprano Sax",
    # "Pipe Organ",
    # "Accordion",
    # "Viola",
    # "Trumpet",
    # "Muted Trumpet",
    # "Oboe",
    # "Clarinet",
    # "Piccolo",
    # "Pan Flute",
    # "Harmonica",
    # "Choir Aahs",
]

C_LIST = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",  # The last three are for OOD
]

roots = [
    "C",
    "G",
    "D",
    "A",
    "E",
    "B",
    "F#",
    "C#",
    "G#",
    "D#",
    "A#",
    "F",
]  # circle of fifths


def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    worker_id = worker_info.id if worker_info else 0
    seed = torch.initial_seed() + worker_id
    seed = seed % (2**32)
    setup_seed(seed)


class InsNotesDataset(IterableDataset):
    def __init__(self, n_segments=90, transform=None, data_type=None, mode="major"):
        """
        data_dir: directory directly containing .npy files

        sr=16000
        """
        self.n_segments = n_segments
        if data_type:
            self.data_type = int(data_type)
        else:
            self.data_type = None
        self.transform = nn.Sequential(
            Spectrogram(n_fft=1024, win_length=1024, hop_length=256),
            MelScale(n_mels=128, sample_rate=16000, f_min=0, f_max=8000, n_stft=513),
        )
        self.mode = mode

    def __iter__(self):
        """
        stream for training
        should make sure each worker has its own generators
        """
        generators = []
        for i in range(len(S_LIST)):
            generators.append(PseudoMelGen(ins_index=i))

        while True:
            i = random.randint(0, len(S_LIST) - 1)  # instrument
            if not self.data_type:
                j = random.randint(0, len(C_LIST) - 4)  # root
            elif isinstance(self.data_type, int):
                j = random.randint(0, self.data_type - 1)
                j = C_LIST.index(roots[j])
            audio, contents = generators[i].gen_melody(
                mel_len=self.n_segments, mode=self.mode, root=j
            )
            audio = torch.tensor(audio)
            audio = self.transform(audio)  # spectrogram
            # audio = audio[
            #     :, :-1, :64
            # ]  # remove the last column because it's always zero. Only for vanilla STFT, also, cut the length to 64 for now
            audio = audio[
                :, :, :32
            ]  # remove the last column because it's always zero. Only for MelSpec, also, cut the length to 64 for now
            audio = torch.log(audio + 1e-6)  # log spectrogram
            audio = audio.unsqueeze(1)  # add channel dimension
            styles = [i for _ in range(self.n_segments)]
            styles = torch.tensor(styles)
            yield audio, contents, styles


class InsNotesTestDataset(Dataset):
    def __init__(
        self, data_dir, n_segments=90, transform=None, data_type=None, mode="major"
    ):
        """
        data_dir: directory directly containing .npy files

        sr=16000
        """
        self.data_dir = data_dir
        self.data_files = glob(os.path.join(data_dir, "*.npy"))
        self.n_segments = n_segments
        self.transform = nn.Sequential(
            Spectrogram(n_fft=1024, win_length=1024, hop_length=256),
            MelScale(n_mels=128, sample_rate=16000, f_min=0, f_max=8000, n_stft=513),
        )
        self.mode = mode

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        """
        get a sample
        """
        data_file = self.data_files[index]
        data_name = os.path.basename(data_file).split(".")[0]
        audio = np.load(data_file)
        audio = torch.tensor(audio)
        audio = self.transform(audio)  # spectrogram
        # audio = audio[
        #     :, :-1, :64
        # ]  # remove the last column because it's always zero. Only for vanilla STFT. also, cut the length to 64 for now
        audio = audio[:, :, :32]  # cut the length to 32 for now
        # randomly select n_segments segments
        assert audio.shape[0] >= self.n_segments
        start = random.randint(0, audio.shape[0] - self.n_segments)
        audio_seg = audio[start : start + self.n_segments, :, :]

        audio_seg = torch.log(audio_seg + 1e-6)  # log spectrogram
        audio_seg = audio_seg.unsqueeze(1)  # add channel dimension
        style = int(data_name.split("_")[0][3:])
        root = int(data_name.split("_")[1][4:])

        # manually generate content labels
        if self.mode == "major":
            p_list = [0, 2, 4, 5, 7, 9, 11]
            while len(p_list) < audio.shape[0]:
                p_list += p_list
            p_list = p_list[start : start + self.n_segments]
        pitches = [(item + root) % 12 for item in p_list]

        contents = torch.tensor(pitches)
        styles = torch.tensor([style for _ in range(self.n_segments)])

        return audio_seg, contents, styles


def get_dataloader(
    batch_size=32,
    n_segments=24,
    # transform=FrequencyMasking(freq_mask_param=15),
    transform=None,
    data_type=None,
    num_workers=4,
    test=False,
    data_dir=None,
    mode="major",
):
    if not test:
        dataset = InsNotesDataset(
            n_segments=n_segments, transform=transform, data_type=data_type, mode=mode
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )
        return iter(dataloader)
    else:
        dataset = InsNotesTestDataset(
            data_dir=data_dir,
            n_segments=n_segments,
            transform=transform,
            data_type=data_type,
            mode=mode,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )
        return dataloader


if __name__ == "__main__":
    train_dl = get_dataloader(
        batch_size=32,
        n_segments=24,
        num_workers=0,
        # test=True,
        # data_dir="../data/insnotes_major_val",
        # mode="major",
    )
    for batch in train_dl:
        print(batch[0].shape, batch[1].shape, batch[2].shape)
        print(batch[1])
        break
