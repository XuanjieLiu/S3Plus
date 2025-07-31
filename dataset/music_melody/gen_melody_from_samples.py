import os
import argparse
from tqdm import tqdm

import numpy as np
import librosa as lr
from soundfile import write

from dataset.music_melody.collect_melody_midi import collect_from_dataset


class PseudoMelGen:
    def __init__(self, ins_index, sr=16000):
        """
        load 12 samples of this instrument (ins_index) from the library
        """
        self.ins_index = ins_index
        self.sr = sr
        self.samples = []
        for i in range(12):
            sample_path = f"dataset/music_melody/samples/wav/ins{str(ins_index).zfill(2)}_{str(i).zfill(2)}.wav"
            sample, _ = lr.load(sample_path, sr=sr)
            self.samples.append(sample)
        self.n_sample_points_per_sample = len(self.samples[0])

    def gen_melody(self, mel_len=12, mode="major", root=0):
        """
        generate a melody of length mel_len
        return the melody (float32), the note boundaries and the pitches
        mode: major or minor or random or chromatic
        root: the root note of the melody (0-11)
        """
        # choose a random scale in "major" (0.3), "minor" (0.3), "dorian" (0.1), "phrygian" (0.1), "lydian" (0.1), "mixolydian" (0.1)
        if mode == "random":
            mode = np.random.choice(
                ["major", "minor", "dorian", "phrygian", "lydian", "mixolydian"],
                p=[0.3, 0.3, 0.1, 0.1, 0.1, 0.1],
            )

        if mode == "major":
            p_list = [0, 2, 4, 5, 7, 9, 11]
        elif mode == "minor":
            p_list = [0, 2, 3, 5, 7, 8, 10]
        elif mode == "dorian":
            p_list = [0, 2, 3, 5, 7, 9, 10]
        elif mode == "phrygian":
            p_list = [0, 1, 3, 5, 7, 8, 10]
        elif mode == "lydian":
            p_list = [0, 2, 4, 6, 7, 9, 11]
        elif mode == "mixolydian":
            p_list = [0, 2, 4, 5, 7, 9, 10]
        elif mode == "chromatic":
            p_list = list(range(12))
        elif (
            mode == "corrupted_chromatic"
        ):  # introduce a small random mistake in the chromatic scale
            p_list = list(range(12))
            # remove a random pitch
            p_list.remove(np.random.choice(p_list))

        melody = []
        pitches = []
        for i in range(mel_len):
            pitch = (p_list[i % len(p_list)] + root) % 12
            note = self.samples[pitch]
            melody.append(note)
            pitches.append(pitch)
        melody = np.array(melody)

        envelope = self.gen_randomize_envelope(n_notes=mel_len)
        melody = melody * envelope
        # normalize the melody
        melody = melody / np.max(np.abs(melody))
        # convert to float32
        melody = melody.astype(np.float32)

        # # get the note boundaries (1) and non-boundaries (0)
        # note_boundaries = np.zeros(len(melody))
        # for i in range(0, mel_len):
        #     note_boundaries[i * len(self.samples[0])] = 1
        # # assert np.sum(note_boundaries) == mel_len

        pitches = np.array(pitches)

        return melody, pitches

    def gen_melody_with_input(self, mel_len=12, input_seq=None, root=0):
        """
        generate a melody of length mel_len
        input_seq: input pitch sequences
        root: the root note of the melody (0-11)
        """

        if input_seq is not None:
            # first transpose input_seq by adding root, then compress range
            input_seq = [(p + root) % 12 for p in input_seq]
            # if longer or shorter, trim or cycle the input sequence
            if len(input_seq) > mel_len:
                input_seq = input_seq[:mel_len]
            elif len(input_seq) < mel_len:
                input_seq = np.tile(input_seq, (mel_len // len(input_seq) + 1))[
                    :mel_len
                ]
            # map the input sequence to the samples
            melody = np.array([self.samples[p] for p in input_seq])

        envelope = self.gen_randomize_envelope(n_notes=mel_len)
        melody = melody * envelope
        # normalize the melody
        melody = melody / np.max(np.abs(melody))
        # convert to float32
        melody = melody.astype(np.float32)

        pitches = np.array(input_seq)

        return melody, pitches

    def gen_randomize_envelope(self, n_notes=12, dont_cat=True):
        """
        Apply a randomized envelope to every note (1s) in the audio.
        The envelope is one of:
        1. linear
        2. Sinusoidal (1/4 period)
        3. Exponential
        The min velocity coefficient is 0.8, and the max is 1.2.
        The max velocity change is 0.2.
        dont_cat: if True, return the envelope for each note separately
        """
        sr = int(self.sr)
        dur_per_note = 1  # note that each note is played for 0.96s, and the second starts at 1.056s. There is a very short overlap between notes.
        if not dont_cat:
            envelope = np.ones(n_notes * sr * dur_per_note)
            start_velocity = 1.0
            end_velocities = np.random.uniform(0.8, 1.2, size=n_notes)
            for i in range(n_notes):
                end_velocity = end_velocities[i]
                curve_type = np.random.choice(["linear", "sinusoidal"])
                if curve_type == "linear":
                    envelope[i * sr * dur_per_note : (i + 1) * sr * dur_per_note] = (
                        np.linspace(start_velocity, end_velocity, sr * dur_per_note)
                    )
                elif curve_type == "sinusoidal":
                    envelope[i * sr * dur_per_note : (i + 1) * sr * dur_per_note] = (
                        np.sin(np.linspace(0, np.pi / 2, sr * dur_per_note))
                        * (end_velocity - start_velocity)
                        + start_velocity
                    )
                # elif curve_type == "exponential":
                #     envelope[i * sr * dur_per_note : (i + 1) * sr * dur_per_note] = np.exp(
                #         np.linspace(
                #             np.log(start_velocity),
                #             np.log(end_velocity),
                #             sr * dur_per_note,
                #         )
                #     )
                start_velocity = end_velocity
            return envelope
        else:
            envelope = []
            for i in range(n_notes):
                start_velocity = 1.0
                end_velocity = np.random.uniform(0.9, 1.1)
                curve_type = np.random.choice(["linear", "sinusoidal"])
                if curve_type == "linear":
                    envelope.append(
                        np.linspace(
                            start_velocity,
                            end_velocity,
                            self.n_sample_points_per_sample,
                        )
                    )
                elif curve_type == "sinusoidal":
                    envelope.append(
                        np.sin(
                            np.linspace(0, np.pi / 2, self.n_sample_points_per_sample)
                        )
                        * (end_velocity - start_velocity)
                        + start_velocity
                    )
                # elif curve_type == "exponential":
                #     envelope.append(
                #         np.exp(
                #             np.linspace(np.log(start_velocity), np.log(end_velocity), sr * dur_per_note)
                #         )
                #     )
            return np.array(envelope)


def gen_directory(save_dir, mode="major", ood=False, styles="all"):
    """
    generate a directory maybe for testing, mode scales
    if ood is True, generate out-of-distribution keys (for now they are A, A#, B)
    """
    S_LIST = [
        "Soprano Sax",
        "Pipe Organ",
        "Accordion",
        "Viola",
        "Trumpet",
        "Muted Trumpet",
        "Oboe",
        "Clarinet",
        "Piccolo",
        "Pan Flute",
        "Harmonica",
        "Choir Aahs",
    ]
    if not styles == "all":
        S_LIST = [S_LIST[0]]

    generators = []
    for i in range(len(S_LIST)):
        generators.append(PseudoMelGen(ins_index=i))

    for round in tqdm(range(10)):
        for i in range(len(S_LIST)):
            if not ood:
                for j in range(12):
                    audio, contents = generators[i].gen_melody(
                        mel_len=90, mode=mode, root=j
                    )
                    # save to npy for now
                    np.save(
                        os.path.join(save_dir, f"ins{i}_root{j}_{round}.npy"), audio
                    )
            else:
                for j in range(9, 12):  # A, A#, B
                    audio, contents = generators[i].gen_melody(
                        mel_len=90, mode=mode, root=j
                    )
                    # save to npy for now
                    np.save(
                        os.path.join(save_dir, f"ins{i}_root{j}_{round}.npy"), audio
                    )
    # # write one sample to wav
    # audio = audio.reshape(-1)
    # write(
    #     os.path.join("./", f"ins{i}_root{j}_{round}.wav"),
    #     audio,
    #     16000,
    # )


def gen_directories_val_ood_spectrum(save_dir, save_name, mode="major", styles="all"):
    """
    generate multiple directories of mode scales, including val and ood sets
    val: from 3 scales to 11 scales
    ood: from 9 scales to 1 scale

    """
    S_LIST = [
        "Soprano Sax",
        "Pipe Organ",
        "Accordion",
        "Viola",
        "Trumpet",
        "Muted Trumpet",
        "Oboe",
        "Clarinet",
        "Piccolo",
        "Pan Flute",
        "Harmonica",
        "Choir Aahs",
    ]
    if not styles == "all":
        S_LIST = [S_LIST[0]]

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
        "B",
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

    generators = []
    for i in range(len(S_LIST)):
        generators.append(PseudoMelGen(ins_index=i))

    for num_vals in range(3, 12):  # from 3 to 11 scales
        save_dir_val = os.path.join(save_dir, save_name + "val" + str(num_vals))
        os.makedirs(save_dir_val, exist_ok=True)
        for round in tqdm(range(10)):  # number of rounds for larger dataset
            for i in range(len(S_LIST)):  # instrument
                for j in range(num_vals):  # in-distribution roots
                    r = C_LIST.index(roots[j])
                    audio, contents = generators[i].gen_melody(
                        mel_len=90, mode=mode, root=r
                    )
                    # save to npy for now
                    np.save(
                        os.path.join(save_dir_val, f"ins{i}_root{r}_{round}.npy"), audio
                    )
        save_dir_ood = os.path.join(save_dir, save_name + "ood" + str(12 - num_vals))
        os.makedirs(save_dir_ood, exist_ok=True)
        for round in tqdm(range(10)):
            for i in range(len(S_LIST)):
                for j in range(num_vals, 12):  # ood roots
                    r = C_LIST.index(roots[j])
                    audio, contents = generators[i].gen_melody(
                        mel_len=90, mode=mode, root=r
                    )
                    # save to npy for now
                    np.save(
                        os.path.join(save_dir_ood, f"ins{i}_root{r}_{round}.npy"), audio
                    )
        print("Generated", save_dir_val, "and", save_dir_ood)


def gen_directory_with_dataset_input(save_dir, data_dir="../data/Nottingham/melody"):
    """
    generate a directory maybe for testing
    """
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

    generators = []
    for i in range(len(S_LIST)):
        generators.append(PseudoMelGen(ins_index=i))

    input_seqs = collect_from_dataset(data_dir=data_dir)

    for k in tqdm(range(len(input_seqs))):
        for round in range(1):
            for i in range(len(S_LIST)):
                for j in range(1):  # # only one root for now for speed
                    audio, contents = generators[i].gen_melody_with_input(
                        mel_len=90, input_seq=input_seqs[k], root=j
                    )
                    # save to npy for now
                    np.save(
                        os.path.join(save_dir, f"ins{i}_root{j}_{str(k).zfill(5)}.npy"),
                        audio,
                    )
    # # write one sample to wav
    # audio = audio.reshape(-1)
    # write(
    #     os.path.join("./", f"ins{i}_root{j}_{str(k).zfill(5)}.wav"),
    #     audio,
    #     16000,
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # os.makedirs("../data/insnotes_major_val", exist_ok=True)
    # gen_directory("../data/insnotes_major_val")
    # print("Generated insnotes_major_val")
    # os.makedirs("../data/insnotes_major_ood", exist_ok=True)
    # gen_directory("../data/insnotes_major_ood", ood=True)
    # print("Generated insnotes_major_ood")

    gen_directory("../data/sax_major_all", ood=False, styles="1")

    gen_directories_val_ood_spectrum(
        save_dir="../data/sax_major_val_ood_spectrum",
        save_name="insnotes_major_",
        mode="major",
        styles="1",
    )

    # os.makedirs("../data/insnotes_nth_val", exist_ok=True)
    # gen_directory_with_dataset_input(
    #     "../data/insnotes_nth_val", data_dir="../data/Nottingham/melody"
    # )
    # gen_directory_with_dataset_input(
    #     "../data/insnotes_0_nth_val", data_dir="../data/Nottingham/melody"
    # )
