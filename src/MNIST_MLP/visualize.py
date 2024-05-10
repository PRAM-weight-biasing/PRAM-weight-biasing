import torch
from torch import tensor
from torchmetrics.classification import MulticlassConfusionMatrix

def accuracy_evo(acc_list: list) -> None:
    pass

def loss_evo(loss_list: list) -> None:
    pass

def confusion_matrix(y_ans: tensor, y_pred: tensor) -> None:
    metric = MulticlassConfusionMatrix(num_classes=10)
    metric.update(y_pred, y_ans)

    fig_, ax_ = metric.plot()