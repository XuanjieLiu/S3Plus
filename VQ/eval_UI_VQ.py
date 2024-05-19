import tkinter
from importlib import reload
import sys
from dataMaker_commonFunc import plot_a_scatter, DOT_POSITIONS
from VQVAE import VQVAE
from tkinter import *
from PIL import Image, ImageTk
from torchvision.utils import save_image
import os
import torch
from torchvision import transforms
from shared import DEVICE

EXP_path = '2023.12.17_multiStyle_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_2_realPair'
SUB_EXP = '1'
MODEL_PATH = 'checkpoint_10000.pt'

EXP_ROOT_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/exp')
RANGE = (3.0, -2.5)
TICK_INTERVAL = 0.05
IMG_PATH = 'eval_decoder_ImgBuffer'
IGM_NAME = IMG_PATH + "/test.png"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

COLORS_TRAIN = ['purple', 'salmon',  'olive', 'blue', 'red', 'green', 'black', 'yellow']
NUMBERS = range(0, 21)
MARKERS = ['o', 'v', 's', 'd', '^', 'X', 'p', 'D']


def find_config():
    exp_path = os.path.join(EXP_ROOT_PATH, EXP_path)
    os.chdir(exp_path)
    sys.path.append(exp_path)
    print(f'Exp path: {exp_path}')
    t_config = __import__('train_config')
    reload(t_config)
    sys.path.pop()
    print(t_config.CONFIG)
    sub_exp_path = os.path.join(exp_path, SUB_EXP)
    os.chdir(sub_exp_path)
    return t_config.CONFIG


def init_vae(config):
    model = VQVAE(config).to(DEVICE)
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
    def __init__(self, win, config):
        os.makedirs(IMG_PATH, exist_ok=True)
        self.vae = init_vae(config)

        # Init decoder eval
        dec_frame = Frame(win)
        dec_frame.pack(side=LEFT)

        self.eval_dec = EvalDecoder(dec_frame, self.vae, config)

        # Init encoder eval
        enc_frame_1 = Frame(win)
        enc_frame_1.pack(side=LEFT)
        self.eval_enc_1 = EvalEncoder(enc_frame_1, self.vae, 1, config, self.eval_dec.update_scales_by_z)
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
        out_z, _ = self.vae.plus(v1, v2)
        self.var_out.set(str(round(out_z.item(), 3)))
        print(f'{round(v1.item(), 3)}, {round(v2.item(), 3)} => {round(out_z.item(), 3)}')


class EvalEncoder:
    def __init__(self, win, vae, idx, config, after_menu_select: callable = None):
        z_s_dim = config['latent_code_2']
        n_emb = config['latent_embedding_1']
        emb_dim = config['embedding_dim']
        self.z_c_dim = n_emb * emb_dim
        self.code_len = self.z_c_dim + z_s_dim
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
        self.after_menu_select = after_menu_select


    def init_z_list(self, win):
        z_var_list = []
        for i in range(self.code_len):
            z_var = tkinter.StringVar()
            z_var.set(str(0.0))
            z_var_list.append(z_var)
            label_text = f'z_c {i+1}: ' if i < self.z_c_dim else f'z_s {i-self.z_c_dim+1}: '
            z_name_label = Label(win, text=label_text)
            z_name_label.grid(row=i, column=0)
            z_value_label = Label(win, textvariable=z_var)
            z_value_label.grid(row=i, column=1)
        return z_var_list

    def update_z_list(self, z):
        for i in range(self.code_len):
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
        plot_a_scatter(DOT_POSITIONS[num], self.img_path, shape, color, num != 0)
        load_img(self.card, self.img_path)
        img_tensor = read_a_data_from_disk(self.img_path).to(DEVICE)
        z = self.vae.batch_encode_to_z(img_tensor)[0][0]
        self.update_z_list(z.cpu().detach())
        self.after_menu_select(z.unsqueeze(0))


class EvalDecoder:
    def __init__(self, win, vae, config):
        self.vae = vae
        z_s_dim = config['latent_code_2']
        self.codebook_size = config['embeddings_num']
        self.z_s_code = self.init_codes(z_s_dim)
        self.n_emb = config['latent_embedding_1']
        self.z_c_code = self.init_codes(self.n_emb)
        scale_frame = Frame(win)
        scale_frame.pack(side=TOP)
        self.z_c_scale_list = self.init_z_c_scale_list(scale_frame)
        self.z_s_scale_list = self.init_z_s_scale_list(scale_frame)
        self.photo = None
        self.label = Label(win)
        self.label.pack(side=BOTTOM)
        self.decoder_config = config['network_config']['enc_dec']

    def update_scales_by_z(self, z):
        idx_z = self.vae.find_indices(z, True, True).cpu().detach()[0]
        z_c = [int(n) for n in str(int(idx_z[0].item()))]
        if len(z_c) == 1:
            z_c = [0, *z_c]
        z_s = idx_z[1:]
        for i in range(self.n_emb):
            self.z_c_code[i] = z_c[i]
            self.z_c_scale_list[i].set(z_c[i])
        for i in range(len(z_s)):
            self.z_s_code[i] = z_s[i].item()
            self.z_s_scale_list[i].set(float(z_s[i].item()))
        self.recon_img()
        self.load_img()

    def on_z_s_scale_move(self, value, index):
        self.z_s_code[index] = float(value)
        self.z_s_scale_list[index].set(float(value))
        self.recon_img()
        self.load_img()

    def on_z_c_scale_move(self, value, index):
        self.z_c_code[index] = int(value)
        self.z_c_scale_list[index].set(int(value))
        self.recon_img()
        self.load_img()


    def recon_img(self):
        img_channel = self.decoder_config['img_channel']
        first_ch_num = self.decoder_config['first_ch_num']
        last_ch_num = first_ch_num * 4
        last_H = self.decoder_config['last_H']
        last_W = self.decoder_config['last_W']
        content_idx = torch.tensor([self.z_c_code]).to(device)
        z_c = self.vae.vq_layer.quantize(content_idx)[0].to(device)
        z_s = torch.tensor(self.z_s_code, dtype=torch.float).to(device)
        z = torch.cat((z_c, z_s), -1).unsqueeze(0)
        #sample = self.vae.decoder(self.vae.fc3(z).view(1, last_ch_num, last_H, last_W))
        sample = self.vae.batch_decode_from_z(z)
        save_image(sample.data.view(1, img_channel, sample.size(2), sample.size(3)), IGM_NAME)

    def load_img(self):
        image = Image.open(IGM_NAME)
        img = image.resize((200, 200))
        # img = image
        self.photo = ImageTk.PhotoImage(img)
        self.label.config(image=self.photo)

    def init_codes(self, code_len):
        codes = []
        for i in range(0, code_len):
            codes.append(0)
        return codes

    def init_z_c_scale_list(self, win):
        scale_list = []
        for i in range(0, self.n_emb):
            scale = Scale(
                win,
                variable=IntVar(value=self.z_c_code[i]),
                command=lambda value, index=i: self.on_z_c_scale_move(value, index),
                from_=0,
                to=self.codebook_size-1,
                resolution=1,
                length=600,
                tickinterval=1
            )
            scale.pack(side=LEFT)
            scale_list.append(scale)
        return scale_list

    def init_z_s_scale_list(self, win):
        scale_list = []
        for i in range(0, len(self.z_s_code)):
            scale = Scale(
                win,
                variable=DoubleVar(value=self.z_s_code[i]),
                command=lambda value, index=i: self.on_z_s_scale_move(value, index),
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
    config = find_config()
    test_ui = TestUI(win, config)
    win.mainloop()
