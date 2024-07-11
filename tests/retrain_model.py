import torch
import random
import os

from torch.utils.data import DataLoader
import torchvision.datasets as dsets
from torchvision.transforms import ToTensor

# import customized files
from network import PruneModel, TrainModel, Vis_Model

### =============================================================

# Setting
random.seed(777)
torch.manual_seed(777)

dir_name = os.getcwd() + '/TestRun/'
# ===========================================
""" need to change """
test_time = "Test_2024-07-02 17:11:46" 
# ===========================================
folder_path = dir_name + test_time
model_name = 'global_pruned_model.pth'


""" for test """ # ===========================

# load model
pruned_model = torch.load(f'{folder_path}/{model_name}')

# Create masks for the weights that are zero
masks = {}
for name, param in pruned_model.named_parameters():
    mask = (param != 0).float()  # Mask where 0 weights are identified
    masks[name] = mask
    
# Register hooks to apply the masks after backward pass
for name, param in pruned_model.named_parameters():
    if name in masks:
        param.register_hook(lambda grad, mask=masks[name]: grad * mask)
  
# Load data - MNIST
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=ToTensor(),  # transform : convert image to tensor. Normalized to 0~1
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=ToTensor(),
                         download=True)

# train_test dataloader
batch_size = 100
train_dataloader = DataLoader(mnist_train, batch_size= batch_size, shuffle=True)
test_dataloader = DataLoader(mnist_test, batch_size= batch_size, shuffle=False)


""" Training """
num_epochs = 30
lr = 0.01

retrain_folder = folder_path + '/retrain'
os.makedirs(retrain_folder, exist_ok=True)

retrain_model = TrainModel(train_dataloader, test_dataloader, pruned_model)
retrain_model.re_MNIST_MLP(lr, num_epochs, retrain_folder, masks)


""" Visualize """
best_model = Vis_Model('best_model.pth', retrain_folder)
best_model.Vis_weight()