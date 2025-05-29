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
    midi_path, sr = args
    wav_path = os.path.splitext(midi_path)[0] + ".wav"
    fs = FluidSynth("../data/FluidR3_GM.sf2", sample_rate=sr)
    fs.midi_to_audio(midi_path, wav_path)
    print("rendered", wav_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_path", type=str, required=True)
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    render_melody_audio((args.midi_path, args.sr))
