from torch import tensor
from torchmetrics.classification import MulticlassConfusionMatrix

import matplotlib.pyplot as plt
import numpy as np

"""=============================================================="""
class vis_model():
    def __init__(self, model, savedir: str) -> None:
        """visualize model, plot training results

        Args:
            model (MLP class object): _description_
            savedir (str): directory for saving the plotted figures
        """
        self.model  = model
        self.savedir= savedir

    def accuracy_evo(self) -> None:
        """Plots accuracy evolution through training epochs

        Returns:
            _type_: None
        """
        plt.figure(figsize=(12, 8))
        
        plt.plot(np.arange(0, len(self.model.acc_list)),
                 [x * 100 for x in self.model.acc_list], 'bs-')
        
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel('Accuracy (%)', fontsize=20)
        
        plt.savefig(f'{self.savedir}/accuracy.png')
        
        return None

    def loss_evo(self) -> None:
        """Plots loss evolution through training epochs

        Returns:
            _type_: None
        """
        plt.figure(figsize=(12, 8))
        
        plt.plot(np.arange(0, len(self.model.loss_list)), 
                 self.model.loss_list, color='black')
        plt.yscale('log')
        
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel('Cross-entropy Loss', fontsize=20)
        
        plt.savefig(f'{self.savedir}/loss.png')
        
        return None

    def confusion_matrix(self, y_ans: tensor, y_pred: tensor) -> None:
        """Plots (0~9) digit confusion matrix of predicted/ground truth

        Args:
            y_ans (tensor): tensor of test dataset ground truth label
            y_pred (tensor): tensor of predicted outputs from test dataset
            savedir (str): directory for saving the plotted figure

        Returns:
            _type_: None
        """
        # create confusion matrix of all 10 digits with torch.metrics library
        metric = MulticlassConfusionMatrix(num_classes=10)
        metric.update(y_pred, y_ans)

        # fig_ is pyplot figure object
        fig_, ax_ = metric.plot()
        fig_.savefig(f'{self.savedir}/confusion_matrix.png')
        
        fig_.clf()
        
        return None
    
    def vis_weight(self, layer=-1) -> None:
        """view distribution of weights in a histogram

        Args:
            layer (int, optional): decide which layer to print.
                                Defaults to -1, which is global case.

        Returns:
            _type_: None
        """
        all_weights = []
        keyword     = 'global' # for determining save file name
        if layer == 1:      # target of interest: fc1 weight
            keyword = 'fc1'
            for name, param in self.model.named_paramters():
                if 'fc1.weight' in name:
                    all_weights.extend(param.detach().numpy().flatten())
        elif layer == 2:    # target of interest: fc2 weight
            keyword = 'fc2'
            for name, param in self.model.named_paramters():
                if 'fc2.weight' in name:
                    all_weights.extend(param.detach().numpy().flatten())
        elif layer == 3:    # target of interest: fc3 weight
            keyword = 'fc3'
            for name, param in self.model.named_paramters():
                if 'fc3.weight' in name:
                    all_weights.extend(param.detach().numpy().flatten())
        else:               # target of interest: global weight
            for name, param in self.model.named_paramters():
                if 'weight' in name:
                    all_weights.extend(param.detach().numpy().flatten())
        
        # plot histogram of targeted weights        
        plt.hist(all_weights, bins=100, alpha=1, color=[0.3, 0.3, 0.8])
        plt.title('Weight Distribution of ', keyword)
        plt.xlabel('Weight')
        plt.ylabel('Frequency')
        plt.grid(True, axis='y')
        
        # save figure and clear figure
        plt.savefig(f'{self.savedir}/{keyword}_weights.png')
        plt.clf()
        
        return None