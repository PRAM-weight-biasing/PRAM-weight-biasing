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

# import customized files
from network import InfModel
import myModule

### =============================================================

# Setting
myModule.fix_seed()
start = time.time()

dir_name = os.getcwd() + '/TestRun/'
# ===========================================
""" need to change """
test_time = "Test_2024-08-01 11:52:18" 
# folder_name = test_time

folder_name = "Test_2024-10-18_13-45_MLP_p0.3"
# "Test_2024-10-18_13-45_MLP_p0.3"
# "Test_2024-10-24_13-35_MLP_p0.4"
# "Test_2024-10-24_13-36_MLP_p0.5"
# "Test_2024-10-24_13-36_MLP_p0.6"
# "Test_2024-10-24_13-36_MLP_p0.7"
# ===========================================
folder_path = dir_name + folder_name
model_name = 'local_pruned_model.pth'

model = torch.load(f'{folder_path}/{model_name}')

# test dataset
transform = transforms.Compose([
            transforms.ToTensor(),   # transform : convert image to tensor. Normalized to 0~1
            transforms.Normalize((0.1307,), (0.3081,))
])

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transform,
                         download=True)

batch_size = 100
test_dataloader = DataLoader(mnist_test, batch_size= batch_size, shuffle=False, num_workers=0)


""" inference accuracy in sw """
n_reps = 10   # Number of inference repetitions.

inf_model = InfModel(model, "mnist")
inf_model.sw_EvalModel(test_dataloader, n_reps)


""" inference accuracy in hw (simulator) """
# convert to aihwkit simulator
inf_model = InfModel(model, "mnist")
analog_model = inf_model.ConvertModel()  # convert s/w model to analog h/w model using aihwkit

# Inference
t_inferences = [0.0, 10.0, 100.0, 1000.0, 3600.0, 10000.0, 86400.0, 1e7, 1e8, 1e9, 1e10, 1e12, 1e15]
n_reps = 10   # Number of inference repetitions.

inf_model.hw_EvalModel(analog_model, test_dataloader, t_inferences, n_reps)



""" onoff ratio test """
# g_minmax_list = [[1000.0, 0.0], [100.0, 0.0],[25.0, 0.0], [10.0, 0.0],
#                  [1000.0, 5.0], [100.0, 5.0],[25.0, 5.0], [10.0, 5.0],
#                  [1000.0, 10.0], [1000.0, 20.0],[1000.0, 50.0], [1000.0, 100.0]]

# for g_list in g_minmax_list:
#     print(f"-------\ng_max,min = {g_list}\n-------")
    
#     inf_model = InfModel(model=model, mode="mnist", g_list=g_list)
#     analog_model = inf_model.ConvertModel()  # convert s/w model to analog h/w model using aihwkit

#     # Inference
#     t_inferences = [0.0, 10.0, 100.0, 1000.0, 3600.0, 10000.0, 86400.0, 1e7, 1e8, 1e9, 1e10, 1e12, 1e15]
#     n_reps = 10   # Number of inference repetitions.

#     inf_model.hw_EvalModel(analog_model, test_dataloader, t_inferences, n_reps)
    
    
""" -------- """

myModule.clear_memory()

# ------------------------------------------------------------------
# measure run-time
sec = time.time()-start
times = str(datetime.timedelta(seconds=sec)) 
short = times.split(".")[0]   # until sec.
print(f"\nruntime : {short} sec\n")