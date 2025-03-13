import torch
import random
import os
import time
import datetime

from torch.utils.data import DataLoader
import torchvision.datasets as dsets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

from Model.PyTorch_CIFAR10.cifar10_models.resnet import resnet18

# import customized files
from network import PruneModel, TrainModel, Vis_Model, InfModel
import myModule

### =============================================================

# Setting
myModule.fix_seed()
start = time.time()

# Load model
dir_name = os.getcwd() + '/TestRun/'
# ===========================================
folder_name = "Test_2024-10-28_15-27_Resnet18_p0.6" 
# ===========================================
folder_path = dir_name + folder_name
model_name = 'local_pruned_model.pth'

pruned_model = torch.load(f'{folder_path}/{model_name}')

# Load data
datatype = 'cifar10'
trainloader, testloader = myModule.set_dataloader(data_type=datatype)

""" Test before fine-tuning """
inf_model = InfModel(pruned_model, datatype)
inf_model.sw_EvalModel(testloader, n_reps=1)


""" Training """
num_epochs = 30
lr = 1e-4

new_folder = folder_path + '/FineTuning'
os.makedirs(new_folder, exist_ok=True)

retrain_model = TrainModel(trainloader, testloader, pruned_model)
retrain_model.cifar_resnet(learning_rate=lr, num_epochs=num_epochs, folder_path=new_folder)


""" Visualize """
best_model = Vis_Model('best_model.pth', new_folder)
best_model.Vis_weight()