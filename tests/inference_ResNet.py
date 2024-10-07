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

# pretrained resnet model from - https://github.com/huyvnphan/PyTorch_CIFAR10.git
from PyTorch_CIFAR10.cifar10_models.resnet import resnet18

# customized files
from network import InfModel
import myModule

# ------------------------------------------------------------------

# Setting
myModule.fix_seed() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start = time.time()

import_model = True  # use pretrained model or not

if import_model==1:
    # Pretrained model
    model = resnet18(pretrained=True)
else:
    # modified model
    dir_name = os.getcwd() + '/TestRun/'
    # ===========================================
    """ need to change """
    test_time = "Test_2024-08-09_prune_0.6" 
    # ===========================================
    folder_path = dir_name + test_time
    model_name = 'global_pruned_model.pth'

    model = torch.load(f'{folder_path}/{model_name}')


# test dataset
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
])

testset = dsets.CIFAR10(root='dataset/',
                        train=False,
                        download=True,
                        transform=transform)
batch_size=200
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2) 

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')



""" inference accuracy in sw """
n_reps = 1   # Number of inference repetitions.

inf_model = InfModel(model, "cifar10")
inf_model.sw_EvalModel(testloader, n_reps)


""" inference accuracy in hw (simulator) """
# convert to aihwkit simulator
inf_model = InfModel(model, "cifar10")
analog_model = inf_model.ConvertModel()  # convert s/w model to analog h/w model using aihwkit

# Inference
# t_inferences = [0.0, 10.0,  100.0, 1000.0, 3600.0, 10000.0, 86400.0, 1e7, 1e8, 1e9, 1e10, 1e12, 1e15]  
t_inferences = [0.0] # test
n_reps = 2  # Number of inference repetitions.

inf_model.hw_EvalModel(analog_model, testloader, t_inferences, n_reps)

""" test 0820 """
# g_minmax_list = [[25.0, 5.0], [25.0, 0.0]]
# for g_list in g_minmax_list:
#     print(f"-------\ng_max,min = {g_list}\n-------")
    
#     inf_model = InfModel(model=model, mode="cifar10", g_list=g_list)
#     analog_model = inf_model.ConvertModel()  # convert s/w model to analog h/w model using aihwkit

#     # Inference
#     t_inferences = [0.0, 10.0, 100.0, 1000.0, 3600.0, 10000.0, 86400.0, 1e7, 1e8, 1e9, 1e10, 1e12, 1e15]  
#     n_reps = 10  # Number of inference repetitions.

#     inf_model.hw_EvalModel(analog_model, testloader, t_inferences, n_reps)
""" -------- """


# ------------------------------------------------------------------
# measure run-time
sec = time.time()-start
times = str(datetime.timedelta(seconds=sec)) 
short = times.split(".")[0]   # until sec.
print(f"\nruntime : {short} sec\n")