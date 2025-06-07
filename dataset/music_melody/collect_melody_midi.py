import os
import argparse
import random
from glob import glob
from tqdm import tqdm
import numpy as np
import pretty_midi


def collect_from_dataset(
    data_dir,
    only_one_octave=False,
    transpose=False,
    portion=1,
):
    """
    Collect melody from xxx dataset.
    This function automatically collects pitch sequences from midi files, applying optional transformations.

    Results returned as a list of lists, where each list contains the pitch sequence of a melody.

    When only_one_octave==True, all notes will be moved into one octave C3-C4.
    When transpose==True, all notes will be transposed to all 12 keys.
    """
    lib = []

    # collect melody from dataset (this works for Wikifonia and other datasets with no subdirectories)
    file_paths = glob(os.path.join(data_dir, "*.mid"))
    # portion
    random.shuffle(file_paths)
    file_paths = file_paths[: int(len(file_paths) * portion)]

    for file_path in tqdm(file_paths):
        file_name = os.path.basename(file_path)
        try:
            midi_data = pretty_midi.PrettyMIDI(file_path, resolution=440)
        except:
            continue
        if len(midi_data.instruments) > 1:
            midi_data.instruments = [midi_data.instruments[0]]
        notes = midi_data.instruments[0].notes
        pitches = [note.pitch for note in notes]

        key = (
            midi_data.key_signature_changes[0].key_number % 12
            if midi_data.key_signature_changes
            else 0
        )
        pitches = [pitch - key for pitch in pitches]
        if only_one_octave:
            # move all pitches into one octave C3-C4
            pitches = [pitch % 12 + 48 for pitch in pitches]

        lib.append(pitches)
    return lib


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/Nottingham/melody")

    args = parser.parse_args()
    data_dir = args.data_dir
    collect_from_dataset(data_dir)
