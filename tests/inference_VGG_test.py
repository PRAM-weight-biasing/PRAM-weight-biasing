import os
import time
import datetime
import random

# torch
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
from torchvision.transforms import ToTensor
import torch.optim as optim

# aihwkit
from aihwkit.nn.conversion import convert_to_analog

# customized files
from network import InfModel
import myModule

"""

This file is for inference test using pretrained neural network.
Performing analog in-memory computing with AIHWKit from IBM.

Model    : VGG-16 (pretrained)
Dataset  : CIFAR-10

"""


# Default setting
myModule.fix_seed()  # fix all the random seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start = time.time()  # calculate running time


# import model and dataset

model = torch.load('./vgg16_cifar10/model_best.pth') #이렇게 하면 안됨. state_dict만 저장되어 있음



# analog model conversion




# inference test




# total running time
sec = time.time()-start
times = str(datetime.timedelta(seconds=sec)) 
short = times.split(".")[0]   # up to sec.
print(f"\n------------> Total running time : {short} sec \n")