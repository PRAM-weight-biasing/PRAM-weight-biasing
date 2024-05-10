import torch
import random
import os

from torch.utils.data import DataLoader
import torchvision.datasets as dsets
from torchvision.transforms import ToTensor

from aihwkit.nn.conversion import convert_to_analog

# import customized files
from network import InfModel

### =============================================================


# Setting
# random.seed(777)
# torch.manual_seed(777)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dir_name = os.getcwd() + '/TestRun/'
# ===========================================
""" need to change """
test_time = "Test_2024-05-07 16:27:42" 
# ===========================================
folder_path = dir_name + test_time
model_name = 'global_pruned_model.pth'


# test dataset
mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=ToTensor(),
                         download=True)

batch_size = 100
test_dataloader = DataLoader(mnist_test, batch_size= batch_size, shuffle=False)

# convert to aihwkit simulator
inf_model = InfModel(model_name, folder_path)
analog_model = inf_model.ConvertModel()  # convert s/w model to analog h/w model using aihwkit

# Inference
t_inferences = [0.0, 3600.0, 86400.0]  # Times to perform infernece.
n_reps = 3   # Number of inference repetitions.
inf_model.EvalModel(analog_model, test_dataloader, t_inferences, n_reps)