import os
import argparse
import random
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

import numpy as np
import librosa as lr
from soundfile import write
from midi2audio import FluidSynth


def render_melody_audio(args):
    midi_path, wav_path, sr = args
    fs = FluidSynth("../data/FluidR3_GM.sf2", sample_rate=sr)
    fs.midi_to_audio(midi_path, wav_path)
    print("rendered", wav_path)


def render_dir(data_dir, save_dir, sr, portion=1):
    os.makedirs(save_dir, exist_ok=True)
    fs = FluidSynth(
        "../data/GeneralUser GS 1.471/GeneralUser GS v1.471.sf2", sample_rate=sr
    )
    midi_paths = glob(os.path.join(data_dir, "*.mid"))
    random.shuffle(midi_paths)
    midi_paths = midi_paths[: int(len(midi_paths) * portion)]
    pool = Pool(processes=8)
    task_args = []
    for midi_path in midi_paths:
        midi_name = os.path.basename(midi_path)
        wav_name = midi_name.replace(".mid", ".wav")
        wav_path = os.path.join(save_dir, wav_name)
        task_args.append([midi_path, wav_path, sr])
    pool.map(render_melody_audio, task_args)


def normalize_dir(data_dir):
    os.makedirs(save_dir, exist_ok=True)
    wav_paths = glob(os.path.join(data_dir, "*.wav"))
    for wav_path in tqdm(wav_paths):
        y_t, sr = lr.load(wav_path, sr=None)
        y_t = y_t / max(abs(y_t)) * 0.9
        write(wav_path, y_t, sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="samples/midi")
    parser.add_argument("--save_dir", type=str, default="samples/wav")
    parser.add_argument("--sr", type=int, default=16000)

    args = parser.parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir
    sr = args.sr

    render_dir(data_dir, save_dir, sr)
    normalize_dir(save_dir)
