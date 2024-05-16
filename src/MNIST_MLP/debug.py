import matplotlib.pyplot as plt
import numpy as np
from torch import tensor
from torchmetrics.classification import MulticlassConfusionMatrix
import copy

from network import MLP

"""=============================================================="""

astr = input("Enter remarks")

print(astr)