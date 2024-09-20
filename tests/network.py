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
# from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.presets import PCMPresetDevice, PCMPresetUnitCell
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
        plt.savefig(f'{folder_path}/graph.png')
        plt.clf()
        

    
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
            if isinstance(module, nn.Linear):
                self.Vis_w_dist(module)
                file_name = myModule.get_unique_filename(self.folder_path, f'weight_dist_{name}', 'png')
                plt.savefig(f'{self.folder_path}/{file_name}')
                plt.clf()
                
                
class InfModel(TrainModel):
    def __init__(self, model, mode: str, g_list: Optional[list] = None):
        # super().__init__()
        self.model = model
        self.eval_fn = self.get_eval_function(mode)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        g_list = [None, None] if g_list is None else g_list
        self.gmax = g_list[0]
        self.gmin = g_list[1]
        
        
    def get_eval_function(self, mode):
        # Define a dictionary mapping modes to evaluation functions
        eval_function_map = {
            "mnist": self.eval_mnist_mlp,
            "cifar10": self.eval_cifar10
        }

        if mode not in eval_function_map:
            raise ValueError(f"Invalid mode: {mode}. Supported modes are: {list(eval_function_map.keys())}")

        return eval_function_map[mode]
        
        
    def SetConfig(self) :
        rpu_config = InferenceRPUConfig()
        rpu_config.device = PCMPresetUnitCell()      # change to paired PCM devices (Gp-Gm)
        # rpu_config.noise_model = TestNoiseModel()   # change to customized noise model
        """ test """
        rpu_config.noise_model = TestNoiseModel(g_max=self.gmax, g_min=self.gmin)  # customized noise model
        """ ----- """
        rpu_config.drift_compensation = None
        # rpu_config.forward.is_perfect=True
        
        """ Weight modifier parameter """
        # rpu_config.modifier.type = WeightModifierType.ADD_NORMAL  # Fwd/bwd weight noise.
        # rpu_config.modifier.std_dev = 0.1
        # rpu_config.modifier.pdrop = 0.05  
        
        return rpu_config
    
    def ConvertModel(self):
        pcm_config = self.SetConfig()
        analog_model = convert_to_analog(self.model, pcm_config)
        
        # setting
        
        return analog_model

    def hw_EvalModel(self, analog_model, test_loader, t_inferences: list, n_reps: int) :
        analog_model.to(self.device)
        analog_model.eval()
        
        inference_accuracy_values = torch.zeros((len(t_inferences), n_reps))

        for t_id, t in enumerate(t_inferences):
            for i in range(n_reps):
                analog_model.drift_analog_weights(t)
                _, test_accuracy = self.eval_fn(analog_model, test_loader)
                inference_accuracy_values[t_id, i] = test_accuracy
                
            print(
                    f"Test set accuracy (%) at t={t}s: \t mean: {inference_accuracy_values[t_id].mean() :.6f}, \t std: {inference_accuracy_values[t_id].std() :.6f}"
                )
            
    def sw_EvalModel(self, test_loader, n_reps: int) :
        self.model.to(self.device)
        self.model.eval()
        
        inference_accuracy_values = torch.zeros(n_reps)

        for i in range(n_reps):
            _, test_accuracy = self.eval_fn(self.model, test_loader)
            inference_accuracy_values[i] = test_accuracy
            
        print(
                f"Test set accuracy (%) in s/w: \t mean: {inference_accuracy_values.mean() :.6f}, \t std: {inference_accuracy_values.std() :.6f}"
            )
            
    def eval_cifar10(self, model, test_loader) -> float:
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
        
    def semitrain_cifar10(self, model, test_loader) -> float:
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