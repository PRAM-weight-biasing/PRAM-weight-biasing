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

from aihwkit_test.customized_noise_pcm import TestNoiseModel


# Instantiate the model
model = nn.Linear(1000, 1)

fixed_weights = torch.linspace(-1, 1, 1000)
model.weight = Parameter(fixed_weights)

# Print the weights and bias to verify
# print("Weights:", model.weight)
# print("Bias:", model.bias)

rpu_config = InferenceRPUConfig()
rpu_config.device = PCMPresetDevice()       # change to paired PCM devices (Gp-Gm)
rpu_config.noise_model = TestNoiseModel()   # change to customized noise model

analog_model = convert_to_analog(model, rpu_config)

# get initial weights
init_weights, _ = analog_model.get_weights()
init_weights_vec = init_weights.reshape(-1)   # convert to (N,1) tensor

# setting inference mode
analog_model.eval() 
t_inf = 1e6

# get weights after t_inf
analog_model.drift_analog_weights(t_inf)
after_weights, _ = analog_model.get_weights()
after_weights_vec = after_weights.reshape(-1)

# delta weight = after = initial weight
weight_change = after_weights - init_weights
weight_change_vec = weight_change.reshape(-1)

# weight change ratio = delta / initial
weight_change_ratio = weight_change_vec / (init_weights_vec)

# nu_drift
nu_vec = -np.log(after_weights_vec/init_weights_vec+1e-20) / np.log(t_inf+20 / 20)


""" --- custom --- """

mu = "t"
sig = "t"

"""----------------"""

# plotting
plt.plot(init_weights_vec, weight_change_vec)
plt.xlabel('initial weight')
plt.ylabel('weight change')
plt.grid(True)
plt.ylim([-0.6,0.6])
plt.savefig(f'figure2/weight_disparity_{mu}_{sig}.png')
plt.clf()

plt.plot(init_weights_vec, weight_change_ratio)
plt.xlabel('initial weight')
plt.ylabel('weight change ratio(= delta / initial)')
plt.grid(True)
plt.ylim([-1,0.2])
plt.savefig(f'figure2/weight_disparity_ratio_{mu}_{sig}.png')
plt.clf()

plt.plot(init_weights_vec, after_weights_vec)
plt.xlabel('initial weight')
plt.ylabel('after weight')
plt.ylim([-1,1])
plt.grid(True)
plt.savefig(f'figure2/after_vs_before_{mu}_{sig}.png')
plt.clf()

plt.plot(init_weights_vec, nu_vec)
plt.xlabel('initial weight')
plt.ylabel('nu_drift')
plt.grid(True)
# plt.ylim([0,0.2])
plt.savefig(f'figure2/nu_drift_{mu}_{sig}.png')
plt.clf()

