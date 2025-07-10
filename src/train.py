# -*- coding: utf-8 -*-

""" train the model """

import os
import sys
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt

# torch related 
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# customized 
import myModule


class TrainModel:
    """ Class for training model """
    
    def __init__(self, model, train_dataloader, test_dataloader, datatype="cifar10"):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.datatype = datatype
        
        
    def train(self, trainloader, epochs=50, lr=0.001):   
        pass  
    
        
    def MNIST_MLP(self, learning_rate: float, num_epochs: int, folder_path) -> None:
                        
        # Define the cost & optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr= learning_rate)
       
        # Initialize parameters
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