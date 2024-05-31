import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.inference import PCMLikeNoiseModel
import matplotlib.pyplot as plt

test_layer = nn.Linear(2000, 1)

linspace_tensor = torch.linspace(-2, 2, 2000)

#forcing_tensor = Parameter(Tensor([[1.2, 0, -1.5, 0],[2, -3, 0, 0]]))
forcing_tensor = Parameter(linspace_tensor)

test_layer.weight = forcing_tensor

rpu_config = InferenceRPUConfig()

print(rpu_config.device)

rpu_config.noise_model = PCMLikeNoiseModel(g_max=500)

analog_model = convert_to_analog(test_layer, rpu_config)

#print(analog_model.get_weights())
a_weights, _ = analog_model.get_weights()
# plt.hist(a_weights.reshape(-1), 1000)
# plt.savefig('before_drift.png')
# plt.clf()

analog_model.eval()
analog_model.drift_analog_weights(1000)
a_after_weights, _ = analog_model.get_weights()
# plt.hist(a_after_weights.reshape(-1), 1000)
# plt.savefig('after_drift.png')
#plt.clf()

analog_model.drift_analog_weights(100000)
a_after_weights, _ = analog_model.get_weights()
# plt.hist(a_after_weights.reshape(-1), 1000)
# plt.savefig('after_drift_longer.png')
#plt.clf()
# plt.clf()

weight_change = a_after_weights - a_weights
print(weight_change)
plt.plot(a_weights.reshape(-1), weight_change.reshape(-1))
plt.savefig('weight_disparity.png')
plt.clf()
after_weight_vec = a_weights.reshape(-1)
weight_change_vec = weight_change.reshape(-1)

weight_change_ratio = weight_change_vec / after_weight_vec

plt.plot(after_weight_vec, weight_change_ratio)
plt.savefig('weight_disparity_ratio.png')
plt.clf()

#print(analog_model.get_weights())