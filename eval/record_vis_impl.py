import os.path
from tkinter import *
from typing import Callable, List
from eval.record_vis import TextRecordVis, ImgRecordVis, EpochBar, DisplayPanel
from load_batch_record import ExpGroup


COMMON_TXT_KEYS = ['loss_ED', 'plus_recon', 'plus_z', 'loss_oper']
COMMON_TXT_ALIAS = ['Self-recon', 'Plus-recon', 'Plus loss', 'Assoc. loss']


def common_recon_img_filter(name: str):
    return name.split('.')[0].isdigit()


def common_recon_img_name2epoch(name: str):
    return int(name.split('.')[0])


def common_enc_img_filter(name: str):
    return "_plus_z_ks_" in name

def common_dec_img_filter(name: str):
    return "_numVis" in name

def common_enc_img_name2epoch(name: str):
    return int(name.split('_')[0])


def exp2callable_list(exp_dir: str):
    def train_record_creator(win: Frame):
        txt_path = os.path.join(exp_dir, 'Train_record.txt')
        text_record_vis = TextRecordVis(
            win,
            vis_name='(Train)',
            record_dir=txt_path,
            keys=COMMON_TXT_KEYS,
            alias=COMMON_TXT_ALIAS
        )
        return text_record_vis

    def eval_record_creator(win: Frame):
        txt_path = os.path.join(exp_dir, 'Eval_record.txt')
        text_record_vis = TextRecordVis(
            win,
            vis_name='(Eval)',
            record_dir=txt_path,
            keys=COMMON_TXT_KEYS,
            alias=COMMON_TXT_ALIAS
        )
        return text_record_vis

    def train_recon_img_creator(win: Frame):
        img_dir = os.path.join(exp_dir, 'TrainingResults')
        recon_img = ImgRecordVis(
            win,
            vis_name="Train Recons",
            img_dir=img_dir,
            name_filter=common_recon_img_filter,
            name2epoch=common_recon_img_name2epoch,
        )
        return recon_img

    def eval_recon_img_creator(win: Frame):
        img_dir = os.path.join(exp_dir, 'EvalResults')
        recon_img = ImgRecordVis(
            win,
            vis_name="Eval Recons",
            img_dir=img_dir,
            name_filter=common_recon_img_filter,
            name2epoch=common_recon_img_name2epoch,
        )
        return recon_img

    def train_enc_img_creator(win: Frame):
        img_dir = os.path.join(exp_dir, 'TrainingResults')
        enc_img = ImgRecordVis(
            win,
            vis_name="Train Enc",
            img_dir=img_dir,
            name_filter=common_enc_img_filter,
            name2epoch=common_enc_img_name2epoch
        )
        return enc_img

    def eval_enc_img_creator(win: Frame):
        img_dir = os.path.join(exp_dir, 'EvalResults')
        enc_img = ImgRecordVis(
            win,
            vis_name="Eval Enc",
            img_dir=img_dir,
            name_filter=common_enc_img_filter,
            name2epoch=common_enc_img_name2epoch
        )
        return enc_img

    def dec_vis_img_creator(win: Frame):
        img_dir = os.path.join(exp_dir, 'EvalResults')
        enc_img = ImgRecordVis(
            win,
            vis_name="Eval Dec",
            img_dir=img_dir,
            name_filter=common_dec_img_filter,
            name2epoch=common_enc_img_name2epoch
        )
        return enc_img

    def accu_txt_creator(win: Frame):
        txt_path = os.path.join(exp_dir, 'plus_eval.txt')
        text_record_vis = TextRecordVis(
            win,
            vis_name='Plus Accuracy',
            record_dir=txt_path,
            keys=['train_accu', 'eval_accu'],
            alias=['Train accu.', 'Eval accu.']
        )
        return text_record_vis

    return [
        train_record_creator,
        eval_record_creator,
        # train_recon_img_creator,
        # eval_recon_img_creator,
        train_enc_img_creator,
        eval_enc_img_creator,
        dec_vis_img_creator,
        accu_txt_creator
    ]


def exp_group2callable_list(exp_group: ExpGroup):
    widget_list = []
    name_list = []
    for sub_exp in exp_group.sub_exps:
        exp_dir = os.path.join(exp_group.exp_path, sub_exp)
        widget_list.append(exp2callable_list(exp_dir))
        names = [exp_group.exp_alias, sub_exp]
        name_list.append(names)
    return widget_list, name_list


def epoch_bar_creator(win: Frame, on_epoch_change: Callable):
    epoch_bar = EpochBar(
        win,
        epoch_start=0,
        epoch_end=80000,
        epoch_tick=200,
        on_epoch_change=on_epoch_change,
    )
    return epoch_bar


def eg_list2panel_input(eg_list: List[ExpGroup]):
    widget_list = []
    name_list = []
    for eg in eg_list:
        widgets, names = exp_group2callable_list(eg)
        widget_list.extend(widgets)
        name_list.extend(names)
    return widget_list, name_list

def run_eval(eg_group):
    win = Tk()
    epoch_vis_creator_list, exp_name_list = eg_list2panel_input(eg_group)
    display_panel = DisplayPanel(
        win,
        exp_name_list,
        epoch_vis_creator_list,
        epoch_bar_creator
    )
    win.mainloop()

eg1 = ExpGroup(
    exp_name='2023.06.09_10vq_Zc[2]_Zs[0]_edim8_plusUnit128.2_encFc128.2_plusS0.3',
    exp_alias='高维度(8 dim),\n 两位数,\n 多冗余(10 tokens),\n (plusS0.3)',
    sub_exp=[19],
    is_load_record=False
)

eg2 = ExpGroup(
    exp_name='2023.06.09_10vq_Zc[2]_Zs[0]_edim8_plusUnit128.2_encFc128.2_plusS1',
    exp_alias='高维度(8 dim),\n 两位数,\n 多冗余(10 tokens),\n (plusS1)',
    sub_exp=[14],
    is_load_record=False
)

eg3 = ExpGroup(
    exp_name='2023.05.26_10vq_Zc[2]_Zs[0]_edim8_plusUnit128.2_encFc128.2_singleS',
    exp_alias='高维度(8 dim),\n 两位数,\n 多冗余(10 tokens),\n (效果好)',
    sub_exp=[10],
    is_load_record=False
)

eg_group = [eg1, eg2, eg3]
if __name__ == '__main__':
    run_eval(eg_group)