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
    def epoch_update(self, epoch: int):
        pass


class ImgRecordVis(EpochVis):
    def __init__(self,
                 frame_win: Tk,
                 vis_name: str,
                 img_dir: str,
                 name_filter: Callable[[str], bool],
                 name2epoch: Callable[[str], int]
                 ):
        self.img_dir = img_dir
        self.name_list = list(filter(name_filter, os.listdir(img_dir)))
        self.available_epoch_list = [name2epoch(name) for name in self.name_list]
        name = Label(frame_win, text=vis_name)
        name.grid(row=0, column=0)
        self.label_var = StringVar()
        self.epoch_label = Label(frame_win, textvariable=self.label_var)
        self.epoch_label.grid(row=1, column=0)
        self.img_label = Label(frame_win)
        self.img_label.grid(row=2, column=0)

    def epoch_update(self, epoch: int):
        if epoch not in self.available_epoch_list:
            pass
        self.label_var.set(f'Epoch {epoch}')
        idx = self.available_epoch_list.index(epoch)
        img_name = self.name_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        load_img(self.img_label, img_path)


class TextRecordVis(EpochVis):
    def __init__(self,
                 frame_win: Frame,
                 vis_name: str,
                 record_dir: str,
                 keys: List[str],
                 alias: List[str],
                 ):
        assert len(keys) == len(alias), "Error: len(keys) != len(alias)"
        self.keys = keys
        self.records = read_record(record_dir)
        self.available_epoch_list = self.records[keys[0]].X
        name = Label(frame_win, text=vis_name)
        name.pack(side=TOP)

        epoch_frame = Frame(frame_win)
        epoch_frame.pack(side=TOP)
        self.epoch_var = StringVar()
        self.epoch_label = Label(frame_win, textvariable=self.epoch_var)

        self.index_list_frame = Frame(frame_win)
        self.index_list_frame.pack(side=TOP)
        l_name_list, l_value_list, self.var_list = self.init_index_list(self.index_list_frame, alias)

    def init_index_list(self, frame_win, alias):
        l_name_list = []
        l_value_list = []
        var_list = []
        for i in range(len(alias)):
            l_name = Label(frame_win, text=f'{alias[i]}: ')
            l_name.grid(row=i, column=0)
            l_name_list.append(l_name)

            var = StringVar(value=str(0.0))
            var_list.append(var)

            l_value = Label(frame_win, textvariable=var)
            l_value.grid(row=i, column=1)
            l_value_list.append(l_value)
        return l_name_list, l_value_list, var_list

    def epoch_update(self, epoch: int):
        if epoch not in self.available_epoch_list:
            pass
        idx = self.available_epoch_list.index(epoch)
        self.epoch_var.set(str(epoch))
        for i in range(len(self.keys)):
            key = self.keys[i]
            value = self.records[key].Y[idx]
            self.var_list[i].set(str(value))


class EpochBar:
    def __init__(self,
                 frame_win,
                 epoch_start: int,
                 epoch_end: int,
                 epoch_tick: int,
                 on_epoch_change: Callable):
        self.epoch_var = IntVar(value=0)
        self.epoch_bar = Scale(
            frame_win,
            orient=HORIZONTAL,
            from_=epoch_start,
            to=epoch_end,
            resolution=epoch_tick,
            tickinterval=epoch_tick,
            command=on_epoch_change
        )
        self.epoch_bar.pack(side=LEFT)


class NamePanel:
    def __init__(self,
                 win,
                 names: List[str],
                 ):
        self.name_labels = []
        for text in names:
            label = Label(win, text=text)
            label.pack(side=TOP)
            self.name_labels.append(label)


class DisplayPanel:
    def __init__(self,
                 win: Frame,
                 exp_name_list: List[List[str]],
                 epoch_vis_creator_list: List[List[Callable[[Frame], EpochVis]]],
                 epoch_bar_creator: Callable[[Frame, Callable], EpochBar],
                 ):
        self.exp_name_list = exp_name_list
        self.epoch_vis_creator_list = epoch_vis_creator_list
        self.epoch_vis_frame = Frame(win)
        self.epoch_vis_frame.pack(side=TOP)
        self.widget_list = self.make_rows()
        epoch_bar_frame = Frame(win)
        epoch_bar_frame.pack(side=TOP)
        self.epoch_bar = epoch_bar_creator(epoch_bar_frame, self.on_epoch_change)

    def make_rows(self):
        widget_list = []
        for i in range(0, len(self.epoch_vis_creator_list)):
            name_frame = Frame(self.epoch_vis_frame)
            name_frame.grid(row=i, column=0)
            name_panel = NamePanel(name_frame, self.exp_name_list[i])
            for j in range(1, len(self.epoch_vis_creator_list[i])):
                epoch_vis_creator = self.epoch_vis_creator_list[i][j]
                widget_frame = Frame(self.epoch_vis_frame)
                widget_frame.grid(row=i, column=j)
                widget = epoch_vis_creator(widget_frame)
                widget_list.append(widget)
        return widget_list

    def on_epoch_change(self, epoch):
        int_epoch = int(epoch)
        for widget in self.widget_list:
            widget.epoch_update(int_epoch)




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
