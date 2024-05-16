import os

import torch
import torchvision.datasets as dsets
from torchvision.transforms import ToTensor

from network import MLP
from visualize import vis_model

"""==================================================="""
# check hardware
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# import and parse datasets
mnist_dset  = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=ToTensor(),
                          download=True)

train_dset  = None
test_dset   = None

# enter list of hyperparameters to split test
list_of_parameters  = []

# enter remarks on current test
print("Enter remarks:")
exp_mark_str    = input()

for i in range(len(list_of_parameters)):
    print(f"Training Start ({i}/{len(list_of_parameters)})")
    print("===============================================\n")
    # initialize network object
    _model  = MLP()
    # train and evaluate network
    param_iter_list = list_of_parameters[i]
    _model.train_model(train_dset, test_dset, param_iter_list)
    # visualize train result
    _vis_module = vis_model(_model, "./results/MNIST_MLP/")
    
    _vis_module.accuracy_evo()  # plot accuracy figure
    _vis_module.loss_evo        # plot loss figure
    _vis_module.confusion_matrix(y_ans, y_pred) # plot confusion matrix

    print("===============================================\n")
