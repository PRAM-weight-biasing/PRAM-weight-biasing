from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 layer fully connected MLP model for MNIST handwritten digit dataset training
        self.model  = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(784, 256, bias=True)),
                        ('relu1', nn.ReLU()),
                        ('fc2', nn.Linear(256, 100, bias=True)),
                        ('relu2', nn.ReLU()),
                        ('fc3', nn.Linear(100, 10, bias=True))
                        ])
                    )
    
    def train_model(self, train_data, test_data, param_list):
        """start training model with loaded data

        Args:
            train_data (torch.dataloader): train dataset
            test_data (torch.dataloader): test dataset
            param_list (list): hyperparameter list (lr, epochs, batch_size, folder_dir)
        """
        # load dataset into object
        self.train_dataloader   = train_data
        self.test_dataloader    = test_data

        # set hyperparameters and miscellaneous informations
        self.lr         = param_list[0]
        self.epochs     = param_list[1]
        self.batch_num  = param_list[2]
        self.save_dir   = param_list[3]

        # initialize training trackers
        self.best_loss  = float('inf')
        self.best_acc   = float('inf')
        self.best_epoch, self.best_lr   = 0, 0
        self.patience   = 5
        self.train_loss_list    = []
        self.test_loss_list     = []
        self.accruacy           = []

        # set cost function and training optimizer
        loss_fn     = nn.CrossEntropyLoss()
        optimizer   = Adam(self.model.parameters(), lr=self.lr)
        scheduler   = ReduceLROnPlateau(optimizer, 'min', patience=self.patience)

        # train and test
        self.model.train()

        for epoch in range(self.epochs):

            # train
            total_train_loss    = 0
            
            for X, Y in self.train_dataloader: # X: input, Y: label
                # reshape image data into batch of 1D vectors (batch_size, 784)
                X       = X.view(-1, 28 * 28)
                y_pred  = self.model(X)
                loss    = loss_fn(y_pred, Y)
                total_train_loss += loss

                optimizer.zero_grad()   # clear gradients of optimizer
                loss.backward()         # calculate gradient
                optimizer.step()        # update weight 1 backprop step

            total_train_loss    /= len(self.train_dataloader)
            self.train_loss_list.append(total_train_loss)

            # test
            self.model.eval()
            with torch.no_grad():
                test_loss   = 0
                total       = 0
                correct     = 0
                
                for X, Y in self.test_dataloader: # X: input, Y: label
                    # reshape image data into collection of 1D vectors
                    X       = X.view(-1, 28 * 28)
                    y_pred  = self.model(X)
                    test_loss += loss_fn(y_pred, Y)

                    digit_pred  = torch.max(y_pred, 1)[1]
                    total       += Y.size(0)
                    correct     = (digit_pred==Y).sum().item()

                test_loss   /= len(self.test_dataloader)
                self.test_loss_list.append(test_loss)
                test_acc    = correct/total
                self.accruacy.append(test_acc)

                # Store best model so far
                if test_acc > self.best_acc:
                    self.best_acc   = test_acc
                    self.best_loss  = test_loss
                    
                    # save model parameters
                    torch.save(self.model.state_dict(), f'{self.save_dir}/best_model_param.pth')
                    self.best_epoch, self.best_lr   = epoch+1, optimizer.param_groups[0]['lr']

                    # save model
                    torch.save(self.model, f'{self.save_dir}/best_model.pth')

                scheduler.step(test_loss)

                # print training results
                print(f"Epoch {epoch+1}/{self.epochs} (learning rate: {optimizer.param_groups[0]['lr']:.1e}),
                       \t\t Train loss: {total_train_loss: .4e}, \tTest accuracy: {test_acc:.2%}")

            # print train final results
            print(f"=======================\n,
                  Best test loss: {self.best_loss:.4e}, Test accuracy: {self.best_acc:.2%} ,
                  at epoch {self.best_epoch}, lr {self.best_lr:.1e},
                  =======================\n")