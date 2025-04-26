import torch
import random
import os
import time
import datetime
import copy

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
myModule.start_timer()

# Load model
dir_name = os.getcwd() + '/TestRun/'
# ===========================================
folder_name = 'Test_2024-10-28_15-26_Resnet18_p0.5'
# ===========================================
folder_path = dir_name + folder_name
model_name = 'local_pruned_model.pth'

pruned_model = torch.load(f'{folder_path}/{model_name}', map_location='cpu')

# Load data
datatype = 'cifar10'
trainloader, testloader = myModule.set_dataloader(data_type=datatype)

# """ Test before fine-tuning """
# inf_model = InfModel(pruned_model, datatype)
# inf_model.sw_EvalModel(testloader, n_reps=1)


""" Training """
# lr =5e-5
lr_list = [1e-6]    # [5e-4, 1e-4, 5e-5, 1e-5, 1e-6]   
num_epochs = 30

# iterate over learning rates
for lr in lr_list:
    myModule.fix_seed()
    new_folder = folder_path + f'/FT_rev1_{lr}_{num_epochs}'
    os.makedirs(new_folder, exist_ok=True)
    print(new_folder)

    model_copy = copy.deepcopy(pruned_model)
    retrain_model = TrainModel(trainloader, testloader, model_copy)
    retrain_model.cifar_neg_reg(learning_rate=1e-4, 
                                num_epochs=num_epochs, 
                                folder_path=new_folder,
                                lambda_val = lr,
                                multiplier_th=1.0,
                                smoothL1_beta=0.5
                                )


    """ Visualize """
    best_model = Vis_Model('best_model.pth', new_folder)
    # best_model.Vis_weight()
    
    myModule.clear_memory()


# another test2
    
folder_name2 = 'Test_2024-10-28_15-27_Resnet18_p0.6'
# ===========================================
folder_path2 = dir_name + folder_name2
pruned_model2 = torch.load(f'{folder_path2}/{model_name}', map_location='cpu')

""" Training """
# lr =5e-5
lr_list2 = [1e-4, 1e-5, 1e-6]       # [5e-4, 1e-4, 5e-5, 1e-5, 1e-6]   
num_epochs = 50

# iterate over learning rates
for lr in lr_list2:
    myModule.fix_seed()
    new_folder = folder_path2 + f'/FT_rev1_{lr}_{num_epochs}'
    os.makedirs(new_folder, exist_ok=True)
    print(new_folder)

    model_copy = copy.deepcopy(pruned_model2)
    retrain_model = TrainModel(trainloader, testloader, model_copy)
    retrain_model.cifar_neg_reg(learning_rate=1e-4, 
                                num_epochs=num_epochs, 
                                folder_path=new_folder,
                                lambda_val = lr,
                                multiplier_th=1.0,
                                smoothL1_beta=0.5
                                )


    """ Visualize """
    best_model = Vis_Model('best_model.pth', new_folder)
    # best_model.Vis_weight()
    
    myModule.clear_memory()
    
    
# another test3
    
folder_name3 = 'Test_2024-10-28_15-32_Resnet18_p0.7'
# ===========================================
folder_path3 = dir_name + folder_name3
pruned_model3 = torch.load(f'{folder_path3}/{model_name}', map_location='cpu')

""" Training """
# lr =5e-5
lr_list3 = [1e-4, 1e-5, 1e-6]       # [5e-4, 1e-4, 5e-5, 1e-5, 1e-6]   
num_epochs = 50

# iterate over learning rates
for lr in lr_list3:
    myModule.fix_seed()
    new_folder = folder_path3 + f'/FT_rev1_{lr}_{num_epochs}'
    os.makedirs(new_folder, exist_ok=True)
    print(new_folder)

    model_copy = copy.deepcopy(pruned_model3)
    retrain_model = TrainModel(trainloader, testloader, model_copy)
    retrain_model.cifar_neg_reg(learning_rate=1e-4, 
                                num_epochs=num_epochs, 
                                folder_path=new_folder,
                                lambda_val = lr,
                                multiplier_th=1.0,
                                smoothL1_beta=0.5
                                )


    """ Visualize """
    best_model = Vis_Model('best_model.pth', new_folder)
    # best_model.Vis_weight()
    
    myModule.clear_memory()




#-------------------------------
myModule.end_timer()