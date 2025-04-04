import torch
import random
import os
import time
import datetime
import numpy as np

from aihwkit.nn.conversion import convert_to_analog
from Model.PyTorch_CIFAR10.cifar10_models.resnet import resnet18

# import customized files
from network import InfModel
import myModule

### =============================================================
# Setting
myModule.fix_seed()
start = time.time()

# trace the imported files in aihwkit folder
# tracelog = myModule.trace()
# tracelog.start_trace()

dir_name = os.getcwd() + '/TestRun/'
# ===========================================
# dir_name = os.getcwd() + '/Model/'

name_list = ["pretrained-model"]
# name_list = ["MLP"]
# name_list = ["Test_2024-10-28_15-15_Resnet18_p0.3"
#              , "Test_2024-10-28_15-22_Resnet18_p0.4"
#              , "Test_2024-10-28_15-26_Resnet18_p0.5"
# #              , "Test_2024-10-24_13-36_MLP_p0.6"
# #              , "Test_2024-10-24_13-36_MLP_p0.7"
#                ]
# ===========================================

model_type = input("Input model type? (1: MLP/2: Resnet18) : ")
print(name_list)
model_name = 'best_model.pth'

# set test dataloader
datatype, testloader = myModule.set_dataloader(model_type)

# iteration
for folder_name in name_list:
    print(f'folder : {folder_name}')
    
    folder_path = dir_name + folder_name
    # model = torch.load(f'{folder_path}/{model_name}')
    model = resnet18(pretrained=True)

    """ inference accuracy in sw """
    n_reps = 10   # Number of inference repetitions.
    inf_model = InfModel(model, datatype)
    # inf_model.sw_EvalModel(testloader, n_reps)


    """ inference accuracy in hw (simulator) """
    # # convert to aihwkit simulator
    # inf_model = InfModel(model, datatype)
    # analog_model = inf_model.ConvertModel()  # convert s/w model to analog h/w model using aihwkit

    # # Inference
    # # t_inferences = [0.0]
    # t_inferences = [0.0, 10.0, 100.0, 1000.0, 3600.0, 10000.0, 86400.0, 1e7, 1e8, 1e9]
    # n_reps = 10   # Number of inference repetitions.
    # inf_model.hw_EvalModel(analog_model, testloader, t_inferences, n_reps)

    # myModule.clear_memory()

""" iteration test (onoff ratio) """
g_minmax_list = [[0.13987,25], [0.19014,25], [0.24878,25], [0.32823,25]]  # [min,max]
# model = torch.load(os.getcwd() + '/Model/MLP/best_model.pth')
model = resnet18(pretrained=True)

for g_list in g_minmax_list:
    print(f"-------\ng_min,max = {g_list}\n-------")
    
    inf_model = InfModel(model=model, mode=datatype, g_list=g_list)
    analog_model = inf_model.ConvertModel()  # convert s/w model to analog h/w model using aihwkit

    # Inference
    t_inferences = [0.0, 10.0, 100.0, 1000.0, 3600.0, 10000.0, 86400.0, 1e7, 1e8, 1e9]  
    n_reps = 10  # Number of inference repetitions.
    inf_model.hw_EvalModel(analog_model, testloader, t_inferences, n_reps)
    
    myModule.clear_memory() # clear memory
    
""" -------- """

# tracing ends
# tracelog.save_trace_results()

# ------------------------------------------------------------------
# measure run-time
sec = time.time()-start
times = str(datetime.timedelta(seconds=sec)) 
short = times.split(".")[0]   # until sec.
print(f"\nruntime : {short} sec\n")