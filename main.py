# Based on:
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
import cv2
import sys
import torch
import random
import numpy as np

seed = 1

random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

from cgan import Data_Manager, CGAN

# --------------------------------------------------------------------------------
if __name__ == '__main__':
    device_id = '3'

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(device_id))


    batch_size = 64

    img_size = 32
    img_shape = [1, img_size, img_size]
    n_classes = 10

    lr = 2e-4
    gen_z_dim = 100

    # Data Manager
    data_manager = Data_Manager(batch_size, img_size)

    # C-VAE
    model = CGAN(data_manager=data_manager, device=device,
                 img_shape=img_shape, n_classes=n_classes,
                 epochs=50,lr=lr, gen_z_dim=gen_z_dim)
    model.fit()

