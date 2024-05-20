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

import aihwkit
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.devices import OneSidedUnitCell

"""==================================================="""
rpu_config      = InferenceRPUConfig()
rpu_config.device = OneSidedUnitCell()
print(rpu_config.device)