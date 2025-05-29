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
    fs = FluidSynth("./dataset/insnotes/soundfonts/FluidR3_GM.sf2", sample_rate=sr)
    fs.midi_to_audio(midi_path, wav_path)
    print("rendered", wav_path)


def render_dir(data_dir, save_dir, sr, portion=1):
    os.makedirs(save_dir, exist_ok=True)
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


def gen_randomize_envelope(sr=16000, dur_per_note=1, n_notes=120):
    """
    Apply a randomized envelope to every note (1s) in the audio.
    The envelope is one of:
    1. linear
    2. Sinusoidal (1/4 period)
    The min velocity coefficient is 0.8, and the max is 1.2.
    The max velocity change is 0.2.
    """
    envelope = np.ones(n_notes * sr * dur_per_note)
    start_velocity = 1.0
    for i in range(n_notes):
        end_velocity = np.random.uniform(
            min(0.8, start_velocity - 0.2), max(1.2, start_velocity + 0.2)
        )
        curve_type = np.random.choice(["linear", "sinusoidal"])
        if curve_type == "linear":
            envelope[i * sr * dur_per_note : (i + 1) * sr * dur_per_note] = np.linspace(
                start_velocity, end_velocity, sr * dur_per_note
            )
        elif curve_type == "sinusoidal":
            envelope[i * sr * dur_per_note : (i + 1) * sr * dur_per_note] = (
                np.sin(np.linspace(0, np.pi / 2, sr * dur_per_note))
                * (end_velocity - start_velocity)
                + start_velocity
            )
        start_velocity = end_velocity
    return envelope


def curve_and_norm_dir(data_dir):
    """
    Add amplitude envelope to the audio files in the directory, and normalize them.
    """
    os.makedirs(save_dir, exist_ok=True)
    wav_paths = glob(os.path.join(data_dir, "*.wav"))
    for wav_path in tqdm(wav_paths):
        y_t, sr = lr.load(wav_path, sr=None)
        envelope = gen_randomize_envelope(sr=sr, dur_per_note=1, n_notes=120)
        pad_one_side = (len(y_t) - len(envelope)) // 2
        envelope = np.pad(envelope, (pad_one_side, pad_one_side), mode="edge")
        y_t = y_t * envelope
        y_t = y_t / max(abs(y_t)) * 0.9
        write(wav_path, y_t, sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/insnotes_midi_",
        help="Directory to load the midi files from",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../data/insnotes_wav_",
        help="Directory to save the generated wav files",
    )
    parser.add_argument("--sr", type=int, default=16000)

    args = parser.parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir
    sr = args.sr

    render_dir(data_dir, save_dir, sr)
    curve_and_norm_dir(save_dir)
