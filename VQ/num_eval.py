import sys
import os
from importlib import reload
from torch.utils.data import DataLoader
from dataloader import SingleImgDataset, load_enc_eval_data
from VQ.VQVAE import VQVAE
from shared import *
import matplotlib.markers
import matplotlib.pyplot as plt
from VQ.eval_common import EvalHelper

matplotlib.use('AGG')



def plot_z_against_label(num_z, num_labels, eval_path=None, eval_helper: EvalHelper = None):
    fig, axs = plt.subplots(1, num_z.size(1), figsize=(num_z.size(1) * 7, 5))
    if num_z.size(1) == 1:
        axs = [axs]
    for i in range(0, num_z.size(1)):
        x = num_labels
        y = num_z[:, i].detach().cpu()
        axs[i].scatter(x, y)
        axs[i].set_title(f'z{i + 1}')
        axs[i].set(xlabel='Num of Points on the card', xticks=range(0, 18))
        if eval_helper is not None:
            eval_helper.draw_scatter_point_line_or_grid(axs[i], i, x, y)
            eval_helper.set_axis(axs[i], i)
        else:
            axs[i].grid(True)

    # for ax in axs.flat:
    #     ax.label_outer()
    if eval_path is None:
        plt.show()
    else:
        plt.savefig(eval_path)
        plt.cla()
        plt.clf()
        plt.close()


def plot_num_position_in_two_dim_repr(num_z, num_labels, result_path=None):
    assert len(num_z[0]) == 2, f"The representation dimension of a number should be two, but got {len(num_z[0])} instead."
    sorted_label = sorted(num_labels)
    sorted_indices = [i[0] for i in sorted(enumerate(num_labels), key=lambda x: x[1])]
    sorted_num_z = [num_z[i] for i in sorted_indices]
    X = [item[0] for item in sorted_num_z]
    Y = [item[1] for item in sorted_num_z]
    for i in range(0, len(num_z)):
        plt.scatter(X[i], Y[i], marker=f'${sorted_label[i]}$', s=60)
    plt.plot(X, Y, linestyle='dashed', linewidth=0.5)
    if result_path is None:
        plt.show()
    else:
        plt.savefig(result_path)
        plt.cla()
        plt.clf()
        plt.close()




class MumEval:
    def __init__(self, config, model_path, data_set_path):
        self.config = config
        dataset = SingleImgDataset(data_set_path)
        self.batch_size = config['batch_size']
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model = VQVAE(config).to(DEVICE)
        if model_path is not None:
            self.reload_model(model_path)
        self.model.eval()

    def reload_model(self, model_path):
        self.model.load_state_dict(self.model.load_tensor(model_path))

    def num_eval_multi_dim(self):
        num_z, num_labels = load_enc_eval_data(
                                    self.loader,
                                    lambda x: self.model.find_indices(
                                          self.model.batch_encode_to_z(x)[0], True
                                    )
        )
        eval_helper = EvalHelper(self.config)
        plot_z_against_label(num_z, num_labels, eval_helper=eval_helper)

    def num_eval_two_dim(self, result_path=None):
        num_z, num_labels = load_enc_eval_data(
            self.loader,
            lambda x:
                self.model.batch_encode_to_z(x)[0]
        )
        num_z = num_z.cpu().detach().numpy()
        plot_num_position_in_two_dim_repr(num_z, num_labels, result_path)



if __name__ == "__main__":
    matplotlib.use('tkagg')
    DATASET_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../dataset/(0,20)-FixedPos-oneStyle')
    EXP_ROOT_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/exp')
    EXP_NAME = '2023.09.25_100vq_Zc[1]_Zs[0]_edim2_[0-20]_plus1024_1'
    SUB_EXP = 16
    CHECK_POINT = 'checkpoint_60000.pt'
    exp_path = os.path.join(EXP_ROOT_PATH, EXP_NAME)
    sys.path.append(exp_path)
    os.chdir(exp_path)
    print(f'Exp path: {exp_path}')
    t_config = __import__('train_config')
    reload(t_config)
    sys.path.pop()
    model_path = os.path.join(exp_path, str(SUB_EXP), CHECK_POINT)
    evaler = MumEval(t_config.CONFIG, model_path, DATASET_PATH)
    evaler.num_eval_two_dim()
