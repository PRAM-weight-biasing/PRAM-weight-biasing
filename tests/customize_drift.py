import pandas as pd
import numpy as np

# torch related methods
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.inference import PCMLikeNoiseModel
import matplotlib.pyplot as plt
from aihwkit.simulator.presets.devices import PCMPresetDevice


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1000, 1) 
        
    def forward(self, x):
        x = self.fc1(x)
        return x

# Instantiate the model
model = SimpleNN()

fixed_weights = torch.linspace(-2, 2, 2000)
model.fc1.weight = Parameter(fixed_weights)

# Print the weights and bias to verify
print("Weights:", model.fc1.weight)
print("Bias:", model.fc1.bias)

rpu_config = InferenceRPUConfig()
rpu_config.device = PCMPresetDevice()