import os
import matplotlib.pyplot as plt
import time

import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
from torchvision.transforms import ToTensor

from network import MNIST_MLP
from visualize import vis_model, vis_accuracy_errbar
from device_loader import DeviceDataLoader, to_device, check_device
from directory_sorter import sort_folders
"""==================================================="""
# check hardware --> edit device for device over-ride
device  = check_device(device='default')
#print(device)

# import and parse datasets
mnist_dset  = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=ToTensor(),
                          download=True)
# split whole dataset into train and test datasets
train_dset, test_dset   = torch.utils.data.random_split(mnist_dset, [50000, 10000])

# enter list of hyperparameters to split test
list_of_parameters  = [[0.001, 5, 'AUTO'],
                       [0.01, 5, 'AUTO']]
batch_size          = 100

# load training data
mnist_train = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
mnist_test  = DataLoader(test_dset, shuffle=False)

# load data into device
mnist_train = DeviceDataLoader(mnist_train, device)
mnist_test  = DeviceDataLoader(mnist_test, device)

# enter remarks on current test
print("Enter remarks:")
exp_mark_str        = input()
list_of_parameters  = sort_folders(list_of_parameters, batch_size, 
                                   exp_mark_str, './results/MNIST_MLP')

start_time  = time.time()
for i in range(len(list_of_parameters)):
    print(f"Training Start ({i+1}/{len(list_of_parameters)})")
    print("===============================================\n")
    accuracy_stat_list  = []
    for seed_num in range(10): # 10 different random seeds
        torch.manual_seed((seed_num+1) * 100)
        print(f'Training with manual torch RNG seed {(seed_num+1)*100}')
        # initialize network object and send to target device
        _model  = MNIST_MLP()
        to_device(_model, device)
        # train and evaluate network
        param_iter_list = list_of_parameters[i][:]
        param_iter_list[2]  = param_iter_list[2] + f'/seed{(seed_num+1) * 100}'
        _model.train_model(mnist_train, mnist_test, param_iter_list)
        # visualize train result
        _vis_module = vis_model(_model, param_iter_list[2])
        
        _vis_module.accuracy_evo()  # plot accuracy figure
        _vis_module.loss_evo()      # plot loss figure
        
        pred_list   = []
        answ_list   = []
        with torch.no_grad():
            for inputs, targets in mnist_test:
                inputs = inputs.view(-1, 28 * 28)
                outputs = _model.model(inputs)
                predict = torch.max(outputs.data, 1)[1]
                pred_list.append(predict)
                answ_list.append(targets)
        # convert into tensors
        pred_tensor = torch.cat(pred_list)
        answ_tensor = torch.cat(answ_list)
        _vis_module.confusion_matrix(answ_tensor, pred_tensor, device) # plot confusion matrix
        accuracy_stat_list.append(_model.accuracy)
        
        if seed_num == 9:
            vis_accuracy_errbar(accuracy_stat_list, list_of_parameters[i][2])
            
        # close all existing figures
        plt.close('all')

    print("===============================================\n")
print("End of test")
end_time    = time.time()
print(f"Total spent time: {(end_time-start_time)/60:.2f} min")