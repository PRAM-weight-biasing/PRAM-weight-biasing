import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
from torchvision.transforms import ToTensor
import random

# import customized files
import myModule
from network import simpleMLP, TrainModel, Vis_Model

### =============================================================


# Setting
random.seed(777)
torch.manual_seed(777)
folder_path = myModule.MakeFolder(0)


""" Data pre-processing """
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
batch_size = 32
train_dataloader = DataLoader(mnist_train, batch_size= batch_size, shuffle=True)
test_dataloader = DataLoader(mnist_test, batch_size= batch_size, shuffle=False)


""" Training """
num_epochs = 100
lr = 0.01

model = simpleMLP()
train_model = TrainModel(train_dataloader, test_dataloader, model)
train_model.MNIST_MLP(lr, num_epochs, folder_path)


""" Visualize """
best_model = Vis_Model('best_model.pth', folder_path)
best_model.Vis_weight()