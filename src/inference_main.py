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

# load the model
imported_model = input("Model type? (1: Pruned /2: FineTuned / 3: Test) : ")

if imported_model == '1':
    model_name = 'local_pruned_model.pth'
elif imported_model == '2':
    model_name = 'FT_0.0001_50/best_model.pth'
elif imported_model == '3':
    model_name = 'test_model.pth'
    
print(f'imported model : {model_name}')

_, testloader = myModule.set_dataloader(data_type="cifar10")


# simulation setting
ideal_io = False
gdc_list = [False] 
g_list = None  # default = None  // [0.1905, 25] 
noise_list = [0, 0]  # pgm, read noise scale respectively
io_res_list = [[6,7], [6,8], [6,9], [7,7], [7,8], [7,9], [8,7], [8,8],[8,9]]  # inp_res, out_res
io_noise_list = [[0.0, 0.0]]   # inp_noise, out_noise