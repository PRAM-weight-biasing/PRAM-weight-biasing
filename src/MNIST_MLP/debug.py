import os
from time import strftime, localtime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
from torchvision.transforms import ToTensor
from torch import Tensor

from network import MNIST_MLP
from visualize import vis_model
from device_loader import DeviceDataLoader
from device_loader import to_device

"""==================================================="""
acc_list = [[0, 1, 2, 3, 4], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]
acc_mean    = np.mean(acc_list, axis=0)
acc_std     = np.std(acc_list, axis=0)
plt.errorbar(np.arange(0, len(acc_list[0])), acc_mean, acc_std,
             fmt='bs-', capsize=4)
plt.savefig('test_errbar.png')