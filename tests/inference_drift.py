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
from Model.PyTorch_CIFAR10.cifar10_models.vgg import vgg16_bn

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

# name_list = ["pretrained-model"]
# name_list = ["MLP"]
name_list = ['Test_2024-10-28_15-15_Resnet18_p0.3'
             , 'Test_2024-10-28_15-22_Resnet18_p0.4'
             , 'Test_2024-10-28_15-26_Resnet18_p0.5'
             , 'Test_2024-10-28_15-27_Resnet18_p0.6'
            #  , "Test_2024-10-24_13-36_MLP_p0.7"
               ]
# ===========================================

model_type = input("Input model type? (1: MLP/2: Resnet or VGG) : ")
print(name_list)
model_name = 'local_pruned_model.pth'

# set test dataloader
if model_type == '1':
    datatype = "mnist"
    # mnist test dataset
    mnist_transform = transforms.Compose([
                transforms.ToTensor(),   # transform : convert image to tensor. Normalized to 0~1
                transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist_test = dsets.MNIST(root='MNIST_data/',
                            train=False,
                            transform=mnist_transform,
                            download=True)

    batch_size = 100
    testloader = DataLoader(mnist_test, batch_size= batch_size, shuffle=False, num_workers=0)

elif model_type == '2':
    datatype = "cifar10"
    # cifar10 test dataset
    cifar10_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
    ])

    cifar10_test = dsets.CIFAR10(root='dataset/',
                            train=False,
                            download=True,
                            transform=cifar10_transform)

    batch_size=200
    testloader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=True, num_workers=2) 


# iteration
for folder_name in name_list:
    print(f'folder : {folder_name}')
    
    folder_path = dir_name + folder_name
    model = torch.load(f'{folder_path}/{model_name}')
    # model = resnet18(pretrained=True)
    # model = vgg16_bn(pretrained=True)

    """ inference accuracy in sw """
    n_reps = 1   # Number of inference repetitions.
    inf_model = InfModel(model, datatype)
    inf_model.sw_EvalModel(testloader, n_reps)


    """ inference accuracy in hw (simulator) """
    # convert to aihwkit simulator
    inf_model = InfModel(model=model, mode=datatype, g_list=[0.1905, 25])
    analog_model = inf_model.ConvertModel()  # convert s/w model to analog h/w model using aihwkit

    # Inference
    t_inferences = [0.0, 10.0, 100.0, 1000.0, 3600.0, 10000.0, 86400.0, 1e7, 1e8, 1e9]
    n_reps = 10   # Number of inference repetitions.
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