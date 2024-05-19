import tkinter

from dataMaker_commonFunc import plot_a_scatter, DOT_POSITIONS
from model import S3Plus, LAST_CN_NUM, LAST_H, LAST_W, IMG_CHANNEL
from train_config import CONFIG
from tkinter import *
from PIL import Image, ImageTk
from torchvision.utils import save_image
import os
import torch
from torchvision import transforms
from shared import DEVICE


RANGE = (3.0, -2.5)
TICK_INTERVAL = 0.05
CODE_LEN = CONFIG['latent_code_1']+CONFIG['latent_code_2']
IMG_PATH = 'eval_decoder-ImgBuffer'
IGM_NAME = IMG_PATH + "/test.png"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_PATH = 'checkpoint_3000.pt'

COLORS_TRAIN = ['purple', 'salmon',  'olive', 'blue']
NUMBERS = range(1, 17)
MARKERS = ['o', 'v', '*', 'd']


def init_codes():
    codes = []
    for i in range(0, CODE_LEN):
        codes.append(0.0)
    return codes


def init_vae():
    model = S3Plus(CONFIG).to(DEVICE)
    model.eval()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"Model is loaded")
    return model


def load_img(spec_label, img_path):
    image = Image.open(img_path)
    img = image.resize((200, 200))
    photo = ImageTk.PhotoImage(img)
    spec_label.config(image=photo)
    spec_label.image = photo


def read_a_data_from_disk(data_path):
    transform = transforms.Compose([transforms.ToTensor()])
    img = Image.open(data_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

class TestUI:
    def __init__(self, win):
        os.makedirs(IMG_PATH, exist_ok=True)
        self.vae = init_vae()

        # Init decoder eval
        dec_frame = Frame(win)
        dec_frame.pack(side=LEFT)
        self.eval_dec = EvalDecoder(dec_frame, self.vae)

        # Init encoder eval
        enc_frame_1 = Frame(win)
        enc_frame_1.pack(side=LEFT)
        self.eval_enc_1 = EvalEncoder(enc_frame_1, self.vae, idx=1)

        # Init plus eval
        plus_frame = Frame(win)
        plus_frame.pack(side=LEFT)
        self.eval_plus = EvalPlus(plus_frame, self.vae)


class EvalPlus:
    def __init__(self, win, vae):
        self.vae = vae
        input_frame = Frame(win)
        input_frame.pack(side=TOP)
        self.var_1, self.var_2 = self.init_input(input_frame)
        output_frame = Frame(win)
        output_frame.pack(side=TOP)
        self.var_out = self.init_output(output_frame)

    def init_input(self, win):
        num_name_1 = Label(win, text='Num 1: ')
        num_name_1.grid(row=0, column=0)
        num_name_2 = Label(win, text='Num 2: ')
        num_name_2.grid(row=1, column=0)
        var_1 = tkinter.StringVar()
        var_1.set(str(0.0))
        var_2 = tkinter.StringVar()
        var_2.set(str(0.0))
        enter_1 = tkinter.Entry(win, textvariable=var_1)
        enter_1.grid(row=0, column=1)
        enter_2 = tkinter.Entry(win, textvariable=var_2)
        enter_2.grid(row=1, column=1)
        return var_1, var_2

    def init_output(self, win):
        bt = tkinter.Button(win, text="Calc", command=self.on_click_calc)
        bt.pack(side=LEFT)
        var_out = tkinter.StringVar()
        var_out.set(str(0.0))
        result = tkinter.Label(win, textvariable=var_out)
        result.pack(side=LEFT)
        return var_out

    def on_click_calc(self):
        v1 = torch.tensor(float(self.var_1.get())).to(DEVICE).unsqueeze(0)
        v2 = torch.tensor(float(self.var_2.get())).to(DEVICE).unsqueeze(0)
        out_z = self.vae.plus(v1, v2)
        self.var_out.set(str(round(out_z.item(), 3)))
        print(f'{round(v1.item(), 3)}, {round(v2.item(), 3)} => {round(out_z.item(), 3)}')


class EvalEncoder:
    def __init__(self, win, vae, idx):
        self.img_name = f"enc_{idx}.png"
        self.img_path = os.path.join(IMG_PATH, self.img_name)
        self.vae = vae
        menu_list_frame = Frame(win)
        menu_list_frame.pack(side=TOP)
        self.num_var, self.color_var, self.shape_var = \
            self.init_menu_list(menu_list_frame)
        self.card = Label(win)
        self.card.pack(side=TOP)
        z_list_frame = Frame(win)
        z_list_frame.pack(side=TOP)
        self.z_var_list = self.init_z_list(z_list_frame)


    def init_z_list(self, win):
        z_var_list = []
        for i in range(CODE_LEN):
            z_var = tkinter.StringVar()
            z_var.set(str(0.0))
            z_var_list.append(z_var)
            z_name_label = Label(win, text=f'z{i+1}: ')
            z_name_label.grid(row=i, column=0)
            z_value_label = Label(win, textvariable=z_var)
            z_value_label.grid(row=i, column=1)
        return z_var_list

    def update_z_list(self, z):
        for i in range(CODE_LEN):
            z_value = str(round(z[i].item(), 3))
            self.z_var_list[i].set(z_value)


    def init_menu_list(self, win):
        # Init labels
        num_label = Label(win, text="Num:")
        num_label.grid(row=0, column=0)
        color_label = Label(win, text="Color:")
        color_label.grid(row=1, column=0)
        shape_label = Label(win, text="Shape:")
        shape_label.grid(row=2, column=0)

        # Init menus
        num_var = tkinter.StringVar()
        num_var.set(str(NUMBERS[0]))
        num_menu = tkinter.OptionMenu(win, num_var, *[str(num) for num in NUMBERS], command=self.on_menu_select)
        num_menu.grid(row=0, column=1)

        color_var = tkinter.StringVar()
        color_var.set(COLORS_TRAIN[0])
        color_menu = tkinter.OptionMenu(win, color_var, *COLORS_TRAIN, command=self.on_menu_select)
        color_menu.grid(row=1, column=1)

        shape_var = tkinter.StringVar()
        shape_var.set(MARKERS[0])
        shape_menu = tkinter.OptionMenu(win, shape_var, *MARKERS, command=self.on_menu_select)
        shape_menu.grid(row=2, column=1)

        return num_var, color_var, shape_var

    def on_menu_select(self, v):
        num = int(self.num_var.get())
        color = self.color_var.get()
        shape = self.shape_var.get()
        plot_a_scatter(DOT_POSITIONS[num], self.img_path, shape, color)
        load_img(self.card, self.img_path)
        img_tensor = read_a_data_from_disk(self.img_path).to(DEVICE)
        z = self.vae.batch_encode_to_z(img_tensor)[1][0].cpu().detach()
        self.update_z_list(z)


class EvalDecoder:
    def __init__(self, win, vae):
        self.vae = vae
        self.code = init_codes()
        scale_frame = Frame(win)
        scale_frame.pack(side=TOP)
        self.scale_list = self.init_scale_list(scale_frame)

        self.photo = None
        self.label = Label(win)
        self.label.pack(side=BOTTOM)

    def on_scale_move(self, value, index):
        self.code[index] = float(value)
        self.scale_list[index].set(float(value))
        self.recon_img()
        self.load_img()

    def recon_img(self):
        codes = torch.tensor(self.code, dtype=torch.float).to(device)
        sample = self.vae.decoder(self.vae.fc3(codes).view(1, LAST_CN_NUM, LAST_H, LAST_W))
        save_image(sample.data.view(1, IMG_CHANNEL, sample.size(2), sample.size(3)), IGM_NAME)

    def load_img(self):
        image = Image.open(IGM_NAME)
        img = image.resize((200, 200))
        # img = image
        self.photo = ImageTk.PhotoImage(img)
        self.label.config(image=self.photo)

    def init_scale_list(self, win):
        scale_list = []
        for i in range(0, CODE_LEN):
            scale = Scale(
                win,
                variable=DoubleVar(value=self.code[i]),
                command=lambda value, index=i: self.on_scale_move(value, index),
                from_=RANGE[0],
                to=RANGE[1],
                resolution=0.01,
                length=600,
                tickinterval=TICK_INTERVAL
            )
            scale.pack(side=LEFT)
            scale_list.append(scale)
        return scale_list


if __name__ == "__main__":
    win = Tk()
    win.title(os.getcwd().split('\\')[-1])
    test_ui = TestUI(win)
    win.mainloop()
