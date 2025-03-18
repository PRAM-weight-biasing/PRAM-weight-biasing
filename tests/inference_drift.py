import torch
import random
import os
import time
import datetime
import numpy as np

from torch.utils.data import DataLoader
import torchvision.datasets as dsets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

from aihwkit.nn.conversion import convert_to_analog
from Model.PyTorch_CIFAR10.cifar10_models.resnet import resnet18
# from Model.PyTorch_CIFAR10.cifar10_models.vgg import vgg16_bn

# import customized files
from network import InfModel
import myModule

### =============================================================
# Setting
# myModule.fix_seed()
start = time.time()

# trace the imported files in aihwkit folder
# tracelog = myModule.trace()
# tracelog.start_trace()

dir_name = os.getcwd() + '/TestRun/'
# dir_name = os.getcwd() + '/Model/'
# ===========================================

# name_list = ["vanilla-MLP"]
name_list = [ 
             'vanilla-Resnet18',
             'Test_2024-10-28_15-15_Resnet18_p0.3',
             'Test_2024-10-28_15-22_Resnet18_p0.4',
             'Test_2024-10-28_15-26_Resnet18_p0.5',
             'Test_2024-10-28_15-27_Resnet18_p0.6',
             'Test_2024-10-28_15-32_Resnet18_p0.7',
               ]
# ===========================================

model_type = input("Input model type? (1: MLP / 2: Resnet18) : ")
imported_model = input("Input model type? (1: Pruned /2: Retrained) : ")

if imported_model == '1':
    model_name = 'local_pruned_model.pth'
elif imported_model == '2':
    model_name = 'FineTuning/best_model.pth'
    
print(f'imported model : {model_name}')
# model_name = 'FineTuning/best_model.pth'  #'local_pruned_model.pth' 'FineTuning/best_model.pth'

# set test dataloader
if model_type == '1':
    datatype = "mnist"
elif model_type == '2':
    datatype = "cifar10"
    
_, testloader = myModule.set_dataloader(data_type=datatype)

# iteration
for folder_name in name_list:
    print(f'folder : {folder_name}')
    
    """ load the model """
    folder_path = dir_name + folder_name
    if 'vanilla' in folder_name:
         model = resnet18(pretrained=True)
         # model = vgg16_bn(pretrained=True)
    else:
        model = torch.load(f'{folder_path}/{model_name}')
    

    """ inference accuracy in sw """
    n_reps = 1   # Number of inference repetitions.
    inf_model = InfModel(model, datatype)
    # inf_model.sw_EvalModel(testloader, n_reps)


    """ inference accuracy in hw (simulator) """
    # simulation setting
    ideal_io = False
    gdc = True
    g_list = None  # default = None  // [0.1905, 25] 
    noise_list = [0, 0]  # pgm, read noise scale respectively
    print(f'--- Ideal-IO:{ideal_io}, GDC:{gdc}, G range={g_list}, noise={noise_list} ---')
    
    inf_model = InfModel(model=model, mode=datatype, g_list=g_list, noise_list=noise_list)
    analog_model = inf_model.ConvertModel(gdc=gdc, ideal_io=ideal_io)  

    # Inference
    t_inferences = [1,                         # 1 sec
                    60,                        # 1 min
                    100,
                    60 * 60,                   # 1 hour
                    24 * 60 * 60,              # 1 day
                    30 * 24 * 60 * 60,         # 1 month
                    12 * 30 * 24 * 60 * 60,    # 1 year
                    36 * 30 * 24 * 60 * 60,    # 3 year
                    1e9,
                    ]
    # [0.0, 10.0, 100.0, 1000.0, 3600.0, 10000.0, 86400.0, 1e6, 1e7, 1e8, 1e9]
    n_reps = 30   # Number of inference repetitions.
    inf_model.hw_EvalModel(analog_model, testloader, t_inferences, n_reps)

    myModule.clear_memory()

# tracing ends
# tracelog.save_trace_results()

# ------------------------------------------------------------------
# measure run-time
sec = time.time()-start
times = str(datetime.timedelta(seconds=sec)) 
short = times.split(".")[0]   # until sec.
print(f"\nruntime : {short} sec\n")