import os

import numpy as np
import pretty_midi


def gen_1sec_midi(
    save_dir, ins_list=[64, 19, 21, 41, 56, 59, 68, 71, 72, 75, 22, 52], n_per_ins=1
):
    """
    Generate easy dataset of midi for training (synthesized data),
    Every file plays the 12 pitches cyclically, each for 1 second, consistently with one instrument, for 2 minutes.
    Every note has a velocity randomly sampled from [60, 120].
    Named like [ins_index]_[file_index].mid

    Instrument indices according to GeneraUser GS v1.43.sf2:
    0=Stereo Grand
    4=Tine Electric Piano
    19=Pipe Organ
    21=Accordian
    22=Harmonica
    40=Violin
    41=Viola
    42=Cello
    52=Choir Aahs
    56=Trumpet
    59=Muted Trumpet
    64=Soprano Sax
    65=Alto Sax
    68=Oboe
    71=Clarinet
    72=Piccolo
    73=Flute
    75=Pan Flute
    76=Blown Bottle
    78=Irish Tin Whistle

    OOD instruments:
    ins_list=[0, 4, 73, 78]

    n_per_ins: number of midi files per instrument
    """
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(ins_list)):
        for j in range(n_per_ins):
            for k in range(13):
                midi_data = pretty_midi.PrettyMIDI(resolution=1000)
                instrument = pretty_midi.Instrument(program=ins_list[i])
                # p_list = np.random.permutation(12)
                note = pretty_midi.Note(
                    velocity=np.random.randint(80, 121),
                    pitch=k + 60,
                    start=0,
                    end=0.96,
                )
                instrument.notes.append(note)
                # in between each note, there is a short pause, which makes each note having 1.056 seconds long, which is 33 hop_lengths (sr=16000, hop_length=512)
                blank = pretty_midi.Note(
                    velocity=0,
                    pitch=0,
                    start=0.96,
                    end=1.056,
                )
                instrument.notes.append(blank)
                midi_data.instruments.append(instrument)
                midi_name = "ins" + str(i).zfill(2) + "_" + str(k).zfill(2) + ".mid"
                midi_data.write(os.path.join(save_dir, midi_name))
                print(f"Generated {midi_name}")


if __name__ == "__main__":
    gen_1sec_midi("samples/midi")
