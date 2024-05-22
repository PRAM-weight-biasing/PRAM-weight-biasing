from aihwkit.simulator.configs import InferenceRPUConfig

# Define the analog model, e.g., by converting a given floating point (FP) model
from torchvision.models import resnet34  # e.g., trained model from torch

rpu_config  = InferenceRPUConfig()

analog_model = convert_to_analog(resnet34(), rpu_config, weight_scaling_omega=1.0)

# [... do hardware-aware (re)-training ...]

# Evaluate the model after programming the weights and driting over a certain period of time
analog_model = AnalogSequential(analog_model).eval()
t_inference = 1000 #desired time of drift in seconds after programming
analog_model.drift_analog_weights(t_inference)

# [... evaluate programmed drifted model ...]