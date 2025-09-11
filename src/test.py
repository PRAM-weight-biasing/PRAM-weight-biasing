""" calculate the power consumption of each model """

# import customized files
import os
import numpy as np
from pathlib import Path
import module.myModule as myModule
from module.model_loader import ModelLoader 
from module.get_power import analyze_array_energy
# from module.get_power import analyze_array_energy_analog
from module.inference import InferenceModel   # <-- InferenceModel import
import torch


# Setting
myModule.start_timer()
myModule.fix_seed()

# load the model
imported_model = '2'  # input("Model type? (1: Pruned /2: FineTuned / 3: Test) : ")
model_dict = ModelLoader.load_models(imported_model)

x = torch.randn(1, 3, 32, 32)
out = []

with torch.no_grad():
    for model_name, model in model_dict.items():
        results = model.conv1(x)
        out.append(results)
                  
print("Diff:", (out[0]- out[1]).abs().mean().item())