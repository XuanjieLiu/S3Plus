import os
import argparse
import random
from copy import deepcopy
from glob import glob
from tqdm import tqdm
import numpy as np
import pretty_midi


def collect_from_dataset(data_dir, save_dir, ins_list=[4, 19, 21, 41, 56, 64, 68, 71, 73, 78], only_one_octave=True, transpose=True, portion=0.02):
    """
    Collect melody from xxx dataset.
    This function automatically changes the instrument of midi according to a list.
    For this study we like instruments with a consistent timbre.
    Instrument indices according to GeneraUser GS v1.43.sf2:
    0=Stereo Grand
    4=Tine Electric Piano
    19=Pipe Organ
    21=Accordian
    40=Violin
    41=Viola
    42=Cello
    56=Trumpet
    64=Soprano Sax
    65=Alto Sax
    68=Oboe
    71=Clarinet
    73=Flute
    78=Irish Tin Whistle

    When only_one_octave==True, all notes will be moved into one octave C4-C5.
    When transpose==True, all notes will be transposed to all 12 keys.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # # collect melody from dataset (this works for POP909)
    # for file_path in tqdm(glob(os.path.join(data_dir, '*/*.mid'))):
    #     file_name = os.path.basename(file_path)
    #     midi_data = pretty_midi.PrettyMIDI(file_path)
    #     midi_data.instruments = [midi_data.instruments[0]]
    #     midi_data.write(os.path.join(save_dir, file_name))

    # collect melody from dataset (this works for Wikifonia and other datasets with no subdirectories)
    file_paths = glob(os.path.join(data_dir, '*.mid'))
    random.shuffle(file_paths)
    file_paths = file_paths[:int(len(file_paths) * portion)]
    for file_path in tqdm(file_paths):
        file_name = os.path.basename(file_path)
        try:
            midi_data = pretty_midi.PrettyMIDI(file_path, resolution=440)
        except:
            continue
        if len(midi_data.instruments) > 1:
            midi_data.instruments = [midi_data.instruments[0]]
            # change instrument
            ins_index = np.random.randint(0, len(ins_list))
            midi_data.instruments[0].program = ins_list[ins_index]
            if not transpose:
                # move to one octave
                if only_one_octave:
                    for note in midi_data.instruments[0].notes:
                        note.pitch = note.pitch % 12 + 60
                # remove the spaces in the file name, and add the instrument index
                file_name = file_name.replace(' ', '')
                file_name = os.path.splitext(file_name)[0] + '_' + str(ins_index).zfill(2) + '.mid'
                midi_data.write(os.path.join(save_dir, file_name))
            elif transpose:
                for i in range(-5, 7):
                    midi_data_transposed = deepcopy(midi_data)
                    # transpose
                    for note in midi_data_transposed.instruments[0].notes:
                        note.pitch += i
                    # move to one octave
                    if only_one_octave:
                        for note in midi_data_transposed.instruments[0].notes:
                            note.pitch = note.pitch % 12 + 60
                    # remove the spaces in the file name, and add the instrument index
                    file_name = file_name.replace(' ', '')
                    file_name_transposed = os.path.splitext(file_name)[0] + '_' + str(ins_index).zfill(2) + '_' + str(i).zfill(2) + '.mid'
                    midi_data_transposed.write(os.path.join(save_dir, file_name_transposed))

                    del midi_data_transposed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../mucodec/data/Wikifonia_midi')
    parser.add_argument('--save_dir', type=str, default='../data/Wikifonia_melody_midi')

    args = parser.parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir
    collect_from_dataset(data_dir, save_dir)