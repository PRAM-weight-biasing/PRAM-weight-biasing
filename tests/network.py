import pandas as pd
import numpy as np
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.datasets as dsets
from torchvision.transforms import ToTensor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

import matplotlib.pyplot as plt
import myModule
    


class simpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256, bias=True)
        self.fc2 = nn.Linear(256, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class TrainModel():
    def __init__(self, train_dataloader, test_dataloader, train_model):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = train_model

    def MNIST_MLP(self, learning_rate: float, num_epochs: int, folder_path) -> None:
                        
        # Define the cost & optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr= learning_rate)
       
        # Initialize parameter
        best_loss = float('inf')  # Initialize with a high number
        best_accuracy = float('inf')
        best_Epoch, best_LR = 0, 0
        patience = 50  # Number of epochs to wait for improvement before stopping
        train_loss_list = []
        test_loss_list = []
        accuracy_list = []

        # # Define the learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience)

        # Train and test
        self.model.train()
        
        for epoch in range(num_epochs):
            
            # train
            total_train_loss = 0

            for X, Y in self.train_dataloader:
                X = X.view(-1, 28 * 28)  # change size to (batch_size, 784)
                y_pred = self.model(X)
                loss = loss_fn(y_pred, Y)
                total_train_loss += loss

                optimizer.zero_grad()   # Initialize the optimizer
                loss.backward()         # calculate gradient
                optimizer.step()        # update weights : w -= lr * w.grad

            total_train_loss /= len(self.train_dataloader)
            train_loss_list.append(total_train_loss)

            # test
            self.model.eval()
            with torch.no_grad():
                test_loss = 0
                total = 0
                correct = 0
                for X, Y in self.test_dataloader:
                    X = X.view(-1, 28 * 28)   # change size to (batch_size, 784)
                    out = self.model(X)
                    test_loss += loss_fn(out, Y)

                    predict = torch.max(out.data, 1)[1]
                    total += Y.size(0)
                    correct += (predict==Y).sum().item()
                    
                test_loss /= len(self.test_dataloader)
                test_loss_list.append(test_loss)
                test_accuracy = correct/total
                accuracy_list.append(test_accuracy)

                # If the test loss improved, save the model
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_accuracy = test_accuracy

                    # save only best model parameter
                    torch.save(self.model.state_dict(), f'{folder_path}/best_model_param.pth')
                    best_Epoch, best_LR = epoch+1, optimizer.param_groups[0]['lr']

                    # Save the model
                    torch.save(self.model, f'{folder_path}/best_model.pth')

                scheduler.step(test_loss)

                # Print info 
                if epoch % 10 == 0:  
                    print(f"Epoch {epoch+1}/{num_epochs}, \tLearning Rate: {optimizer.param_groups[0]['lr']:.1e}, \tTrain Loss: {total_train_loss:.6f}, \tTest Loss: {test_loss:.6f}, \tTest Accuracy : {test_accuracy:.2%}")
        
        print(f"\n Best test loss: {best_loss:.6f}, Test accuracy: {best_accuracy:.2%}, Epoch: {best_Epoch}, LR: {best_LR:.2e}\n")

    
    def CIFAR_ResNet(self, learning_rate: float, num_epochs: int, folder_path) -> None:
        None


    def LossFunction(self, loss_type: str, y_true, y_pred) -> None:
        mse =  nn.MSELoss()(y_true, y_pred)
        smoothL1 = nn.SmoothL1Loss()(y_true, y_pred)
        crossEntropy = torch.nn.CrossEntropyLoss()(y_true, y_pred)

        if loss_type == 'MSE':
            lossfunc = mse
        elif loss_type == "SmoothL1":
            lossfunc = smoothL1
        elif loss_type == "CE":
            lossfunc = crossEntropy
        else: None

        return lossfunc
    

class Vis_Model():
    def __init__(self, model_name: str, folder_path: str):
        best_model = torch.load(f'{folder_path}/{model_name}')
        self.model = best_model
        self.folder_path = folder_path
    
    def Vis_weight(self) -> None :
        all_weights = []

        for name, param in self.model.named_parameters():
            if 'weight' in name:
                all_weights.extend(param.detach().numpy().flatten())
                
        plt.hist(all_weights, bins=30, alpha=1)
        plt.title('Weight Distribution')
        plt.xlabel('Weight value')
        # plt.xlim([-4,4])
        plt.ylabel('Frequency')
        plt.grid(True, axis='y')
        plt.savefig(f'{self.folder_path}/weight_distribution.png')
        
        # plot in log scale
        plt.yscale('log')
        plt.savefig(f'{self.folder_path}/weight_distribution(log).png')
        plt.clf()  
        

class PruneModel():
    def __init__(self, prune_percent: float, model_name: str, folder_path: str):
        self.folder_path = folder_path
        self.premodel = torch.load(f'{self.folder_path}/{model_name}')
        self.prune_percent = prune_percent


    def local_pruning(self) -> None :
        # Apply pruning to each linear layer
        for name, module in self.premodel.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, 'weight', amount= self.prune_percent)
                prune.remove(module, 'weight')  # fix pruned weights
        # Save the model
        torch.save(self.premodel, f'{self.folder_path}/local_pruned_model.pth')


    def global_pruning(self) -> None :
        # Apply pruning to all parameters
        parameters_to_prune = (
            (self.premodel.fc1, 'weight'),
            (self.premodel.fc2, 'weight'),
            (self.premodel.fc3, 'weight'),
        )

        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount= self.prune_percent)

        for name, module in self.premodel.named_modules():
            if isinstance(module, nn.Linear):
                prune.remove(module, 'weight') # fix pruned weights

        # Save the model
        torch.save(self.premodel, f'{self.folder_path}/global_pruned_model.pth')
                
    def cal_sparsity(self, weights) -> float :
        # weights : tensor
        total_params = weights.numel()  # Total number of parameters
        zero_params = torch.sum(weights == 0).item()  # Number of zeroed parameters
        return zero_params / total_params  # Sparsity ratio

    def cal_local_sparsity(self, pruned_model: str) -> list:
        pruned_model = torch.load(f'{self.folder_path}/{pruned_model}')
        # Check sparsity in each layer
        sparsity_list = []
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear):
                sparsity = self.cal_sparsity(module.weight)
                sparsity_list.append(sparsity)
                print(f"{name} sparsity: {sparsity:.2%}")

        return sparsity_list

    def cal_global_sparsity(self, pruned_model: str) -> float:
        pruned_model = torch.load(f'{self.folder_path}/{pruned_model}')
        total_params = 0
        zero_params = 0
        # Iterate over all modules and sum total and zeroed-out parameters
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear):
                total_params += module.weight.numel()
                zero_params += torch.sum(module.weight == 0).item()
        global_sparsity = zero_params / total_params  # Sparsity ratio

        print(f"Global Sparsity: {global_sparsity:.2%}")
        return global_sparsity
    
    def Vis_w_dist(self, weights) -> None:
        all_weights = []
        for name, param in weights.named_parameters():
            if 'weight' in name:
                all_weights.extend(param.detach().numpy().flatten())
            
        plt.hist(all_weights, bins=100, alpha=1)
        plt.title('Weight Distribution')
        plt.xlabel('Weight value')
        plt.ylabel('Frequency')
        plt.grid(True, axis='y')
        

    def Vis_global_w(self) -> None:
        self.Vis_w_dist(self.premodel)
        file_name = myModule.get_unique_filename(self.folder_path, 'weight_dist_global', 'png')
        plt.savefig(f'{self.folder_path}/{file_name}')
        plt.clf()  

    def Vis_local_w(self) -> None:
        for name, module in self.premodel.named_modules():
            if isinstance(module, nn.Linear):
                self.Vis_w_dist(module)
                file_name = myModule.get_unique_filename(self.folder_path, f'weight_dist_{name}', 'png')
                plt.savefig(f'{self.folder_path}/{file_name}')
                plt.clf()