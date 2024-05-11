import matplotlib.pyplot as plt
import numpy as np
from torch import tensor
from torchmetrics.classification import MulticlassConfusionMatrix
import copy

from network import MLP

"""=============================================================="""

ex_net = MLP()
copiednet = copy.deepcopy(ex_net)
#print(ex_net._modules)
for name, para in copiednet.named_parameters():
    if 'fc1.weight' in name:
        print(1)
        print(name)
