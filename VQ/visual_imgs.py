import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import torch
import numpy as np
matplotlib.use('AGG')


class VisImgs:
    def __init__(self, save_path):
        self.save_path = save_path
        self.gt_a = torch.ones(3, 10, 10)
        self.gt_b = torch.ones(3, 10, 10)
        self.gt_c = torch.ones(3, 10, 10)
        self.recon_a = torch.ones(3, 10, 10)
        self.recon_b = torch.ones(3, 10, 10)
        self.recon_c = torch.ones(3, 10, 10)
        self.plus_c = torch.ones(3, 10, 10)


    def arrange_tensor(self, input: torch.Tensor, need_permute=True):
        tensor = input.permute(1, 2, 0) if need_permute else input
        return np.array(tensor.cpu().detach(), dtype=float)

    def save_img(self, name):
        fig, axs = plt.subplots(3, 3)
        axs[0, 0].imshow(self.arrange_tensor(self.gt_a))
        axs[0, 0].set_title("GT a")

        axs[0, 1].imshow(self.arrange_tensor(self.gt_b))
        axs[0, 1].set_title("GT b")

        axs[0, 2].imshow(self.arrange_tensor(self.gt_c))
        axs[0, 2].set_title("GT c")

        axs[1, 0].imshow(self.arrange_tensor(self.recon_a))
        axs[1, 0].set_title("Recon a")

        axs[1, 1].imshow(self.arrange_tensor(self.recon_b))
        axs[1, 1].set_title("Recon b")

        axs[1, 2].imshow(self.arrange_tensor(self.recon_c))
        axs[1, 2].set_title("Recon c")

        axs[2, 2].imshow(self.arrange_tensor(self.plus_c))
        axs[2, 2].set_title("plus c")


        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        save_path = os.path.join(self.save_path, name)
        plt.savefig(save_path)
        plt.cla()
        plt.clf()
        plt.close()



