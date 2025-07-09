# -*- coding: utf-8 -*-

""" inference tests over time to evaluate the impact of resistance drift """

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from model.PyTorch_CIFAR10.cifar10_models.resnet import resnet18

# import customized files
import myModule
from network import InfModel


# Setting
myModule.start_timer()
myModule.fix_seed()

dir_name = os.getcwd() + '/TestRun/'

name_list = [ 
            'vanilla-Resnet18',
            'Resnet18_p0.3',
            # 'Resnet18_p0.4',
            # 'Resnet18_p0.5',
            'Resnet18_p0.6',
            # 'Resnet18_p0.7',
            # 'Resnet18_p0.8',
            'Resnet18_p0.9',
               ]
