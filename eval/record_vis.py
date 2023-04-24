import tkinter
from tkinter import *
from typing import Callable, List
from PIL import Image, ImageTk
import os
from abc import ABC, abstractmethod

from loss_counter import read_record


def load_img(label: tkinter.Label, img_path: str):
    image = Image.open(img_path)
    img = image.resize((200, 200))
    photo = ImageTk.PhotoImage(img)
    label.config(image=photo)
    label.image = photo


class EpochVis(ABC):
    @abstractmethod
    def on_epoch_change(self, epoch: int):
        pass


class ImgRecordVis(EpochVis):
    def __init__(self,
                 frame_win: Tk,
                 img_dir: str,
                 name_filter: Callable[[str], bool],
                 name2epoch: Callable[[str], int]
                 ):
        self.img_dir = img_dir
        self.name_list = list(filter(name_filter, os.listdir(img_dir)))
        self.available_epoch_list = [name2epoch(name) for name in self.name_list]
        self.label_var = StringVar()
        self.epoch_label = Label(frame_win, textvariable=self.label_var)
        self.epoch_label.grid(row=0, column=0)
        self.img_label = Label(frame_win)
        self.img_label.grid(row=1, column=0)

    def on_epoch_change(self, epoch: int):
        if epoch not in self.available_epoch_list:
            pass
        self.label_var.set(f'Epoch {epoch}')
        idx = self.available_epoch_list.index(epoch)
        img_name = self.name_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        load_img(self.img_label, img_path)


class TextRecordVis(EpochVis):
    def __init__(self,
                 frame_win: Tk,
                 record_dir: str,
                 keys: List[str],
                 alias: List[str],
                 ):
        assert len(keys) == len(alias), "Error: len(keys) != len(alias)"
        float_vars = [DoubleVar() for n in keys]
        records = read_record(record_dir)
        self.available_epoch_list = records[keys[0]].X




if __name__ == '__main__':
    dir = "D:/Projects/Gus Xia/S3Plus/VQ/exp/2023.04.20_100vq_Zc[1]_Zs[0]_edim8_plusUnit128.1_encFc128.1_singleS_plusOnE/1/EvalResults"

    win = Tk()


    def name_filter(name: str):
        return name.split('.')[0].isdigit()

    def name2epoch(name: str):
        return int(name.split('.')[0])


    irv = ImgRecordVis(win, dir, name_filter, name2epoch)
    print(irv.name_list)
    irv.epoch_var.set(100)
    irv.epoch_var.set(200)
