import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Optional

# torch related methods
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

# customized methods
import myModule

# aihwkit related methods
from aihwkit.simulator.configs import (
    InferenceRPUConfig,
    WeightNoiseType,
    WeightClipType,
    WeightModifierType,
)
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.presets import PCMPresetDevice, PCMPresetUnitCell

# custmized noise model
from aihwkit_test.customized_noise_pcm import TestNoiseModel


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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = train_model.to(self.device)
        
    def MNIST_MLP(self, learning_rate: float, num_epochs: int, folder_path) -> None:
                        
        # Define the cost & optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr= learning_rate)
       
        # Initialize parameter
        best_loss = float('inf')  # Initialize with a high number
        best_accuracy = 0
        best_Epoch, best_LR = 0, 0
        patience = 5  # Number of epochs to wait for improvement before stopping
        train_loss_list = []
        test_loss_list = []
        accuracy_list = []

        # Define the learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience)

        # Train and test
        self.model.train()
        
        for epoch in range(num_epochs):
            
            # train
            total_train_loss = 0

            for X, Y in self.train_dataloader:
                X = X.view(-1, 28 * 28).to(self.device)  # change size to (batch_size, 784)
                Y = Y.to(self.device)
                y_pred = self.model(X)
                loss = loss_fn(y_pred, Y)
                total_train_loss += loss.item()

                optimizer.zero_grad()   # Initialize the optimizer
                loss.backward()         # calculate gradient
                optimizer.step()        # update weights : w -= lr * w.grad

            total_train_loss /= len(self.train_dataloader)
            train_loss_list.append(total_train_loss)

            # test
            self.model.eval()
            test_loss, test_accuracy = self.eval_mnist_mlp(self.model, self.test_dataloader)
            test_loss_list.append(test_loss)
            accuracy_list.append(test_accuracy)

            # If the accuracy improved, save the model
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_loss = test_loss

                # save only best model parameter
                torch.save(self.model.state_dict(), f'{folder_path}/best_model_param.pth')
                best_Epoch, best_LR = epoch+1, optimizer.param_groups[0]['lr']

                # Save the model
                torch.save(self.model, f'{folder_path}/best_model.pth')

            scheduler.step(test_loss)

            # Print info 
            if epoch % 5 == 0:  
                print(f"Epoch {epoch+1}/{num_epochs}, \tLearning Rate: {optimizer.param_groups[0]['lr']:.1e}, \tTrain Loss: {total_train_loss:.6f}, \tTest Loss: {test_loss:.6f}, \tTest Accuracy : {test_accuracy:.2f}%")
        
        print(f"\n test loss: {best_loss:.6f}, Best test accuracy: {best_accuracy:.2f}%, Epoch: {best_Epoch}, LR: {best_LR:.2e}\n")
        self.Vis_accuracy(num_epochs, train_loss_list, test_loss_list, accuracy_list, folder_path)


    def eval_mnist_mlp(self, model, test_loader) -> float:
        model.eval()
        
        with torch.no_grad():
            test_loss = 0
            total = 0
            correct = 0
            for inputs, targets in test_loader:
                inputs = inputs.view(-1, 28*28).to(self.device)  # change size to (batch_size, 784)
                targets = targets.to(self.device)
                outputs = model(inputs)
                test_loss += nn.CrossEntropyLoss()(outputs, targets)

                predict = torch.max(outputs.data, 1)[1]
                total += targets.size(0)
                correct += (predict==targets).sum().item()
            
            test_loss /= len(test_loader)
            test_accuracy = 100* (correct / total)
                
            return test_loss.item(), test_accuracy
        
    def re_MNIST_MLP(self, learning_rate: float, num_epochs: int, folder_path:str, masks:dict) -> None:
                               
        # Define the cost & optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr= learning_rate)
       
        # Initialize parameter
        best_loss = float('inf')  # Initialize with a high number
        best_accuracy = 0
        best_Epoch, best_LR = 0, 0
        patience = 5  # Number of epochs to wait for improvement before stopping
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
                X = X.view(-1, 28 * 28).to(self.device)  # change size to (batch_size, 784)
                Y = Y.to(self.device)
                y_pred = self.model(X)
                loss = loss_fn(y_pred, Y)
                total_train_loss += loss.item()

                optimizer.zero_grad()   # Initialize the optimizer
                loss.backward()         # calculate gradient
                self.apply_masks(self.model, masks)  # Apply the masks to keep zero weights frozen
                optimizer.step()        # update weights : w -= lr * w.grad

            total_train_loss /= len(self.train_dataloader)
            train_loss_list.append(total_train_loss)

            # test
            self.model.eval()
            test_loss, test_accuracy = self.eval_mnist_mlp(self.model, self.test_dataloader)
            test_loss_list.append(test_loss)
            accuracy_list.append(test_accuracy)

            # If the accuracy improved, save the model
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_loss = test_loss

                # save only best model parameter
                torch.save(self.model.state_dict(), f'{folder_path}/best_model_param.pth')
                best_Epoch, best_LR = epoch+1, optimizer.param_groups[0]['lr']

                # Save the model
                torch.save(self.model, f'{folder_path}/best_model.pth')

            scheduler.step(test_loss)

            # Print info 
            if epoch % 5 == 0:  
                print(f"Epoch {epoch+1}/{num_epochs}, \tLearning Rate: {optimizer.param_groups[0]['lr']:.1e}, \tTrain Loss: {total_train_loss:.6f}, \tTest Loss: {test_loss:.6f}, \tTest Accuracy : {test_accuracy:.2f}%")
        
        print(f"\n test loss: {best_loss:.6f}, Best test accuracy: {best_accuracy:.2f}%, Epoch: {best_Epoch}, LR: {best_LR:.2e}\n")
        self.Vis_accuracy(num_epochs, train_loss_list, test_loss_list, accuracy_list, folder_path)
        
    def apply_masks(self, model, masks: dict):
        # Function to apply masks during the backward pass
        for name, param in model.named_parameters():
            if name in masks:
                param.grad *= masks[name]
    
    def Vis_accuracy(self, epochs, train_loss: list, test_loss: list, accuracy_list: list, folder_path: str) -> None:
        # Visualize loss and accuracy per epoch

        plt.figure(figsize=(12,4))
        
        # convert GPU tensors to CPU
        train_loss = [t.cpu().numpy() if torch.is_tensor(t) else t for t in train_loss]
        test_loss = [t.cpu().numpy() if torch.is_tensor(t) else t for t in test_loss]
        accuracy_list = [t.cpu().numpy() if torch.is_tensor(t) else t for t in accuracy_list]

        # plot loss vs. epoch
        plt.subplot(1,3,1)
        plt.plot(np.arange(0,epochs), train_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Train Loss')

        plt.subplot(1,3,2)
        plt.plot(np.arange(0,epochs), test_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Test Loss')

        # plot accuracy vs. epoch
        plt.subplot(1,3,3)
        plt.plot(np.arange(0,epochs), accuracy_list)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig(f'{folder_path}/ChartPerEpoch.png')
        plt.clf()
    
    def Vis_loss_split(self, epochs, ce_loss_list, custom_loss_list, test_loss_list, acc_list, folder_path: str):
        # Visualize separated loss and accuracy
        ce_loss_list = [t.cpu().numpy() if torch.is_tensor(t) else t for t in ce_loss_list]
        custom_loss_list = [t.cpu().numpy() if torch.is_tensor(t) else t for t in custom_loss_list]
        test_loss_list = [t.cpu().numpy() if torch.is_tensor(t) else t for t in test_loss_list]
        acc_list = [t.cpu().numpy() if torch.is_tensor(t) else t for t in acc_list]
        
        
        # Visualize separated loss and accuracy
        plt.figure(figsize=(12,4))

        # --- First subplot: Loss curves with two y-axis
        ax1 = plt.subplot(1,3,1)
        ax2 = ax1.twinx()  # Create a twin axis sharing the same x-axis

        ax1.plot(np.arange(0,epochs), ce_loss_list, label='Cross Entropy', color='blue')
        ax2.plot(np.arange(0,epochs), custom_loss_list, label='Custom Reg', color='orange', linestyle='--')

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cross Entropy Loss', color='blue')
        ax2.set_ylabel('Custom Reg Loss', color='orange')

        # Add legends separately
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # --- Second subplot: Test Loss
        plt.subplot(1,3,2)
        plt.plot(np.arange(0,epochs), test_loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Test Loss')

        # --- Third subplot: Accuracy
        plt.subplot(1,3,3)
        plt.plot(np.arange(0,epochs), acc_list)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.tight_layout()
        plt.savefig(f'{folder_path}/ChartPerEpoch_split.png')
        plt.clf()
    
    @staticmethod
    def freeze_batchnorm(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()  # prevent BN stat updates
            module.weight.requires_grad = False
            module.bias.requires_grad = False
    
    def cifar_resnet(self, learning_rate: float, num_epochs: int, folder_path) -> None:
        for _, param in self.model.named_parameters():
            param.requires_grad = True  # train all Conv & FC layers
                                    
        # Define the loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr= learning_rate, weight_decay=5e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)  
        
        # Data loader
        from myModule import set_dataloader
        trainloader, testloader = set_dataloader(data_type='cifar10')
        
        # Initialize parameters
        best_loss = float('inf')  # Initialize with a high number
        best_acc = 0
        best_Epoch, best_LR = 0, 0
        train_loss_list = []
        test_loss_list = []
        acc_list = []
        
        # Train and test
        self.model.train()
        self.model.apply(self.freeze_batchnorm)  # fix batchnorm layers
        
        for epoch in range(num_epochs):
            
            # Train
            total_train_loss = 0

            for batch_idx, (inputs, targets) in enumerate(trainloader):  # debug ; self.train_dataloader
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                total_train_loss += loss.item()                
                
                optimizer.zero_grad()   # Initialize the optimizer
                loss.backward()         # calculate gradient
                optimizer.step()        # update weights : w -= lr * w.grad
                
            total_train_loss /= len(trainloader) # debug ; self.train_dataloader
            train_loss_list.append(total_train_loss)
            
            # Test
            self.model.eval()
            test_loss, test_acc = self.eval_cifar10(self.model, testloader) # debug ; self.test_dataloade
            test_loss_list.append(test_loss)
            acc_list.append(test_acc)
            
            # If accuracy is too low, don't update scheduler
            if test_acc > 10:
                scheduler.step(test_loss)
            
            # If the accuracy improved, save the model
            if test_acc > best_acc:
                best_acc = test_acc
                best_loss = test_loss
                
                # save only best model parameter
                torch.save(self.model.state_dict(), f'{folder_path}/best_model_param.pth')
                best_Epoch, best_LR = epoch+1, optimizer.param_groups[0]['lr']

                # Save the model
                torch.save(self.model, f'{folder_path}/best_model.pth')

            # scheduler.step(test_loss)
            
            # Print info 
            if epoch % 5 == 0:  
                print(f"Epoch {epoch+1}/{num_epochs}, \tLearning Rate: {optimizer.param_groups[0]['lr']:.1e}, \tTrain Loss: {total_train_loss:.6f}, \tTest Loss: {test_loss:.6f}, \tTest Accuracy : {test_acc:.2f}%")
        
        print(f"\n test loss: {best_loss:.6f}, Best test accuracy: {best_acc:.2f}%, Epoch: {best_Epoch}, LR: {best_LR:.2e}\n")
        self.Vis_accuracy(num_epochs, train_loss_list, test_loss_list, acc_list, folder_path)
    
    
    def custom_loss(self, model, lambda_val: float, multiplier_th: float):

        total_reg = torch.tensor(0., device=next(model.parameters()).device, requires_grad=True)

        for param in model.parameters():
            if param.requires_grad:
                if param.numel() == 0:
                    continue  # skip empty tensors

                max_weight = param.abs().max()
                threshold = multiplier_th * max_weight

                # threshold 이하 weight만 선택
                mask = (param.abs() < threshold)
                selected_weights = param[mask]

                if selected_weights.numel() > 0:
                    
                    # SmoothL1 Loss 
                    smoothL1_beta = 0.5
                    smoothL1_loss = F.smooth_l1_loss(
                        selected_weights,
                        torch.zeros_like(selected_weights),
                        reduction='sum',  # 전체 합산
                        beta=smoothL1_beta          # 부드러움 조정 (작을수록 0 근처에 민감)
                    )
                    
                    # Inverse Loss
                    epsilon=1e-2
                    inverse_loss = torch.sum(1.0 / (param ** 2 + epsilon))
                    smooth_inverse_loss = torch.sum(1.0 / torch.sqrt(param ** 2 + epsilon))
                    
                    # Quadratic Loss
                    const = 10.0
                    quad_loss = -torch.sum(const*((param**2)-0.16)**2)
                    
                    total_reg = total_reg + quad_loss

        return lambda_val * total_reg
    
    
    def cifar_neg_reg(self, learning_rate: float, num_epochs: int, folder_path, 
                      lambda_val: float, multiplier_th= 1.0) -> None:
        """_summary_

        Args:
            learning_rate (float): 
            num_epochs (int): 
            folder_path (_type_): directory to save the model
            lambda_val (float): coefficient of negative L2 regularization
            multiplier_th (float): threshold multiplier to adjust regularization
        """
        for _, param in self.model.named_parameters():
            param.requires_grad = True  # train all Conv & FC layers
                                    
        # Define the loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr= learning_rate, weight_decay=0)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)  
        
        # Data loader
        from myModule import set_dataloader
        trainloader, testloader = set_dataloader(data_type='cifar10')
        
        # Initialize parameters
        best_loss = float('inf')  # Initialize with a high number
        best_acc = 0
        best_Epoch, best_LR = 0, 0
        train_loss_list = []
        test_loss_list = []
        acc_list = []
        ce_loss_list = []
        custom_loss_list = []
           
        # Train and test
        self.model.train()
        self.model.apply(self.freeze_batchnorm)  # fix batchnorm layers
        
        for epoch in range(num_epochs):
            
            # Train
            total_train_loss = 0
            total_ce_loss = 0
            total_custom_loss = 0

            for batch_idx, (inputs, targets) in enumerate(trainloader):  # debug ; self.train_dataloader
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                ce_loss = loss_fn(outputs, targets)
                
                # add custom loss term
                custom_loss = self.custom_loss(
                    self.model,
                    lambda_val=lambda_val,   # coefficient of customized regularization
                    multiplier_th=multiplier_th,  
                )
                loss = ce_loss + custom_loss   # awary from zero

                total_train_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_custom_loss += custom_loss.item()
                
                    
                optimizer.zero_grad()   # Initialize the optimizer
                loss.backward()         # calculate gradient
                optimizer.step()        # update weights : w -= lr * w.grad
                
            total_train_loss /= len(trainloader) # debug ; self.train_dataloader
            train_loss_list.append(total_train_loss)
            
            total_ce_loss /= len(trainloader)
            total_custom_loss /= len(trainloader)
            ce_loss_list.append(total_ce_loss)
            custom_loss_list.append(total_custom_loss)
            
            # Test
            self.model.eval()
            test_loss, test_acc = self.eval_cifar10(self.model, testloader) # debug ; self.test_dataloade
            test_loss_list.append(test_loss)
            acc_list.append(test_acc)
            
            # If accuracy is too low, don't update scheduler
            if test_acc > 10:
                scheduler.step(test_loss)
            
            # If the accuracy improved, save the model
            if test_acc > best_acc:
                best_acc = test_acc
                best_loss = test_loss
                
                # save only best model parameter
                torch.save(self.model.state_dict(), f'{folder_path}/best_model_param.pth')
                best_Epoch, best_LR = epoch+1, optimizer.param_groups[0]['lr']

                # Save the model
                torch.save(self.model, f'{folder_path}/best_model.pth')

            # scheduler.step(test_loss)
            
            # Print info 
            if epoch % 5 == 0:  
                print(f"Epoch {epoch+1}/{num_epochs}, \tLearning Rate: {optimizer.param_groups[0]['lr']:.1e}, \tTrain Loss: {total_train_loss:.6f}, \tTest Loss: {test_loss:.6f}, \tTest Accuracy : {test_acc:.2f}%")
        
        print(f"\n test loss: {best_loss:.6f}, Best test accuracy: {best_acc:.2f}%, Epoch: {best_Epoch}, LR: {best_LR:.2e}\n")
        self.Vis_accuracy(num_epochs, train_loss_list, test_loss_list, acc_list, folder_path)
        self.Vis_loss_split(num_epochs, ce_loss_list, custom_loss_list, test_loss_list, acc_list, folder_path)
               
            
    def eval_cifar10(self, model, test_loader) -> float:
        # model.to(self.device)
        self.model.eval()

        total = 0
        correct = 0
        test_loss = 0
        
        with torch.no_grad():
            # model.eval()
            
            for data in test_loader:
                images, targets = data[0].to(self.device), data[1].to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                test_loss += nn.CrossEntropyLoss()(outputs, targets)
                        
            test_accuracy = 100* (correct / total)
            test_loss /= len(test_loader)
                
            return test_loss, test_accuracy
    

class Vis_Model():
    def __init__(self, model_name: str, folder_path: str):
        best_model = torch.load(f'{folder_path}/{model_name}')
        self.model = best_model
        self.folder_path = folder_path
    
    def Vis_weight(self) -> None :
        all_weights = []

        for name, param in self.model.named_parameters():
            if 'weight' in name:
                all_weights.extend(param.detach().cpu().numpy().flatten())
                
        plt.figure(figsize=(8,6))
        plt.hist(all_weights, bins=200, alpha=1)
        plt.title('Weight Distribution')
        plt.xlabel('Weight value')
        # plt.xlim([-4,4])
        plt.ylabel('Frequency')
        plt.grid(True, axis='y')
        plt.savefig(f'{self.folder_path}/w_dist.png')
        
        # plot in log scale
        plt.yscale('log')
        plt.savefig(f'{self.folder_path}/w_dist(log).png')
        plt.clf()  
        

class PruneModel():
    def __init__(self, model_name: str, folder_path: str):
        self.folder_path = folder_path
        self.model_name = model_name
        
        self.premodel = self.load_model()

    def load_model(self):
        try:     # for model 
            premodel = torch.load(f'{self.folder_path}/{self.model_name}')
        except:  # for imported model from web
            premodel = self.model_name
        
        return premodel

    def apply_local_pruning(self, amount: float) -> None :
        # Apply pruning to each linear layer
        for name, module in self.premodel.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                prune.l1_unstructured(module, 'weight', amount= amount)
                prune.remove(module, 'weight')  # fix pruned weights
        # Save the model
        torch.save(self.premodel, f'{self.folder_path}/local_pruned_model.pth')


    def apply_global_pruning(self, amount: float) -> None :
        # Apply pruning to all parameters       
        parameters_to_prune = []
        for name, module in self.premodel.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                parameters_to_prune.append((module, 'weight'))

        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount= amount)

        for name, module in self.premodel.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                prune.remove(module, 'weight') # fix pruned weights

        # Save the model
        torch.save(self.premodel, f'{self.folder_path}/global_pruned_model.pth')
                
    def cal_sparsity(self, weights) -> float :
        # weights : tensor
        total_params = weights.numel()  # Total number of parameters
        zero_params = torch.sum(weights == 0).item()  # Number of zeroed parameters
        return zero_params / total_params  # Sparsity ratio

    def cal_local_sparsity(self, pruned_model: str) -> list:
        model = torch.load(f'{self.folder_path}/{pruned_model}')
        # Check sparsity in each layer
        sparsity_list = []
        print(f'\n--- Local sparsity of {str(pruned_model)} ---')
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                sparsity = self.cal_sparsity(module.weight)
                sparsity_list.append(sparsity)
                print(f"{name} sparsity: {sparsity:.2%}")

        return sparsity_list

    def cal_global_sparsity(self, pruned_model: str) -> float:
        model = torch.load(f'{self.folder_path}/{pruned_model}')
        total_params = 0
        zero_params = 0
        # Iterate over all modules and sum total and zeroed-out parameters
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                total_params += module.weight.numel()
                zero_params += torch.sum(module.weight == 0).item()
        global_sparsity = zero_params / total_params  # Sparsity ratio

        print(f'\n--- Global sparsity of {str(pruned_model)} ---')
        print(f"Global Sparsity: {global_sparsity:.2%}")
        return global_sparsity
    
    def Vis_w_dist(self, weights) -> None:
        all_weights = []
        for name, param in weights.named_parameters():
            if 'weight' in name:
                all_weights.extend(param.detach().cpu().numpy().flatten())
            
        plt.hist(all_weights, bins=200, alpha=1)
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
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                self.Vis_w_dist(module)
                file_name = myModule.get_unique_filename(self.folder_path, f'weight_dist_{name}', 'png')
                plt.savefig(f'{self.folder_path}/{file_name}')
                plt.clf()
                
                
class InfModel(TrainModel):
    def __init__(self, model, mode: str, g_list: Optional[list] = None, noise_list: Optional[list]=None):
        # super().__init__()
        self.model = model
        self.eval_fn = self.get_eval_function(mode)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        g_list = [None, None] if g_list is None else g_list
        self.gmin = g_list[0]
        self.gmax = g_list[1]
        
        noise_list = [None, None] if noise_list is None else noise_list
        self.pgm_noise = noise_list[0]
        self.read_noise = noise_list[1]
        
    def get_eval_function(self, mode):
        # Define a dictionary mapping modes to evaluation functions
        eval_function_map = {
            "mnist": self.eval_mnist_mlp,
            "cifar10": self.eval_cifar10
        }

        if mode not in eval_function_map:
            raise ValueError(f"Invalid mode: {mode}. Supported modes are: {list(eval_function_map.keys())}")

        return eval_function_map[mode]
        
        
    def SetConfig(self, gdc:bool, ideal_io:bool) :
        rpu_config = InferenceRPUConfig()
        rpu_config.device = PCMPresetUnitCell()      # change to paired PCM devices (Gp-Gm)
        rpu_config.noise_model = TestNoiseModel(
            g_max=self.gmax, 
            g_min=self.gmin, 
            prog_noise_scale=self.pgm_noise, 
            read_noise_scale=self.read_noise, 
            )  # customized noise model
        rpu_config.mapping.weight_scaling_omega = 1.0  # 넣으면 non-ideal io에서도 conversion success!!!
        
        # global drift compensation
        if gdc == True: pass
        elif gdc == False:
            rpu_config.drift_compensation = None   # apply GDC or not
            
        # ideal io
        if ideal_io == True:
            rpu_config.forward.is_perfect=True     # io parameters
        elif ideal_io == False: pass

        
        return rpu_config
    
    
    def SetConfig_io(
        self, 
        gdc: bool, 
        ideal_io: bool = False,
        inp_res_bit: float = 7, 
        inp_noise: float = 0.01,            # small input Gaussian noise (std-dev)
        out_res_bit: float = 9, 
        out_noise: float = 0.06,            # output Gaussian noise (std-dev)
        ):
        
        rpu_config = InferenceRPUConfig()
        rpu_config.device = PCMPresetUnitCell()      # paired PCM devices (Gp-Gm)
        rpu_config.noise_model = TestNoiseModel(
            g_max=self.gmax, 
            g_min=self.gmin, 
            prog_noise_scale=self.pgm_noise, 
            read_noise_scale=self.read_noise, 
            )  # customized noise model
        rpu_config.mapping.weight_scaling_omega = 1.0  
        
        # global drift compensation
        if gdc == True: pass
        elif gdc == False:
            rpu_config.drift_compensation = None   # apply GDC or not
            
        # ideal io
        if ideal_io == True:
            rpu_config.forward.is_perfect=True     # io parameters
        elif ideal_io == False: pass

        
        """ IO parameter settings """
        rpu_config.forward = IOParameters(
            is_perfect=False,

            # === DAC (Input side) ===
            inp_bound=1.0,                           # DAC input range: [-1, 1]
            inp_res= 1.0 / (2**inp_res_bit - 2),     # n-bit DAC quantization
            inp_noise= inp_noise,
            # inp_sto_round=False,          # enable stochastic rounding in DAC
            # inp_asymmetry=0.0,            # 1% asymmetry in pos/neg DAC signal

            # === ADC (Output side) ===
            out_bound=12.0,                         # ADC saturation limit (max current)
            out_res= 1.0 / (2**out_res_bit - 2),    # n-bit DAC quantization      
            out_noise= out_noise,         
            # out_noise_std=0.1,             # 10% std variation across outputs
            # out_sto_round=False,            # enable stochastic rounding in ADC
            # out_asymmetry=0.005,           # 0.5% asymmetry in negative pass output

            # === Bound & Noise management (recommended for analog) : as default setting ===
            # bound_management=BoundManagementType.ITERATIVE,
            # noise_management=NoiseManagementType.ABS_MAX,

            # === 기타 이상성 제거 : As default setting===
            # w_noise=0.0,                   # no weight noise
            # w_noise_type=WeightNoiseType.NONE,
            # ir_drop=0.0,
            # out_nonlinearity=0.0,
            # r_series=0.0
            
            # from example (if needed)
            # out_res = -1.0  # Turn off (output) ADC discretization.
            # w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
            # w_noise = 0.02  # Short-term w-noise.       
        )

        return rpu_config
    
    def ConvertModel(self, gdc:bool, ideal_io:bool):
        # fix seed for reproducibility during mapping
        myModule.fix_seed(seed=42)  
        
        pcm_config = self.SetConfig(gdc=gdc, ideal_io=ideal_io)
        analog_model = convert_to_analog(self.model, pcm_config)
        
        return analog_model
    
    def ConvertModel_io(
        self, 
        gdc:bool, 
        ideal_io:bool=False,
        inp_res_bit: float = 7, 
        inp_noise: float = 0.0,            
        out_res_bit: float = 9, 
        out_noise: float = 0.0,            
        ):
        # fix seed for reproducibility during mapping
        myModule.fix_seed(seed=42)  
        
        pcm_config = self.SetConfig_io(
            gdc=gdc, 
            ideal_io=ideal_io,
            inp_res_bit=inp_res_bit, 
            inp_noise=inp_noise,            
            out_res_bit=out_res_bit, 
            out_noise=out_noise,           
            )
        analog_model = convert_to_analog(self.model, pcm_config)
        
        return analog_model
    
    
    def hw_EvalModel(self, analog_model, test_loader, t_inferences: list, n_reps: int) -> list :
        """_summary_

        Args:
            analog_model (_type_): _description_
            test_loader (_type_): _description_
            t_inferences (list): _description_
            n_reps (int): number of repetition in fixed analog model

        Returns:
            list: accuracy results
        """
        
        analog_model.to(self.device)
        analog_model.eval()
        
        inference_accuracy_values = torch.zeros((len(t_inferences), n_reps))
        results = []
                
        for t_id, t in enumerate(t_inferences):
            print("[DEBUG] t_inf :", t)
            
            for i in range(n_reps): 
                # fix seed for reproducibility when applying the gaussian noise
                current_seed = np.random.randint(0, 99999)  # 0에서 99999 사이의 랜덤 정수
                print("[DEBUG] Generated Seed:", current_seed)
                torch.manual_seed(current_seed)
                torch.cuda.manual_seed(current_seed)
                            
                # inference
                with torch.no_grad():
                    analog_model.drift_analog_weights(t)
                    _, test_accuracy = self.eval_fn(analog_model, test_loader)
                    inference_accuracy_values[t_id, i] = test_accuracy
                    print("[DEBUG] Accuracy:", test_accuracy)
                
            mean_acc = inference_accuracy_values[t_id].mean().item()
            std_acc = inference_accuracy_values[t_id].std().item()
                        
            print(
                    f"Test set accuracy (%) at t={t}s: \t mean: {mean_acc:.6f}, \t std: {std_acc :.6f}"
                )
            
            results.append([t, mean_acc, std_acc])
            
        return results
    
    def hw_EvalModel_single(self, analog_model, test_loader, t_inferences: list, seed: int, n_reps=1) -> list :
        
        analog_model.to(self.device)
        analog_model.eval()
        
        inference_accuracy_values = torch.zeros((len(t_inferences), n_reps)) 
        results = []
                
        for t_id, t in enumerate(t_inferences):
            # fix seed for reproducibility when applying the gaussian noise
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            
            analog_model.drift_analog_weights(t)
            for i in range(n_reps):              
                # inference
                analog_model.drift_analog_weights(t)
                _, test_accuracy = self.eval_fn(analog_model, test_loader)
                inference_accuracy_values[t_id, i] = test_accuracy
                            
            results.append([t, test_accuracy])
            
        return results
            
    def sw_EvalModel(self, test_loader, n_reps: int) :
        self.model.to(self.device)
        self.model.eval()
        
        inference_accuracy_values = torch.zeros(n_reps)
        myModule.fix_seed()
        
        for i in range(n_reps):
            _, test_accuracy = self.eval_fn(self.model, test_loader)
            inference_accuracy_values[i] = test_accuracy
        
        mean_acc = inference_accuracy_values.mean().item()
        std_acc = inference_accuracy_values.std().item()
            
        print(
                f"Test set accuracy (%) in s/w: \t mean: {mean_acc :.6f}, \t std: {std_acc :.6f}"
            )
            
    def __xeval_cifar10(self, model, test_loader) -> float:
        # Move model to the correct device
        model.to(self.device)
        
        total = 0
        correct = 0
        
        with torch.no_grad():
            model.eval()
            
            for data in test_loader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                        
            test_accuracy = 100* (correct / total)
                
            return total, test_accuracy
        
    def __xsemitrain_cifar10(self, model, test_loader) -> float:
        # Move model to the correct device
        model.to(self.device)
        
        total = 0
        correct = 0
        
        model.train()
        
        for data in test_loader:
            images, labels = data[0].to(self.device), data[1].to(self.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
                    
        test_accuracy = 100* (correct / total)
        
        print(
                f"train mode accuracy (%) : \t mean: {test_accuracy :.6f}"
            )
            
        # return total, test_accuracy