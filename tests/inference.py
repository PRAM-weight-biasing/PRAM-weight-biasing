import torch
import random
import os
import time
import datetime

from torch.utils.data import DataLoader
import torchvision.datasets as dsets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

from aihwkit.nn.conversion import convert_to_analog

# import customized files
from network_test_07 import InfModel

### =============================================================


# Setting
# random.seed(777)
# torch.manual_seed(777)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start = time.time()


dir_name = os.getcwd() + '/TestRun/'
# ===========================================
""" need to change """
test_time = "Test_2024-08-01 11:52:18/retrain" 
# ===========================================
folder_path = dir_name + test_time
model_name = 'best_model.pth'

model = torch.load(f'{folder_path}/{model_name}')

# test dataset
transform = transforms.Compose([
            transforms.ToTensor(),   # transform : convert image to tensor. Normalized to 0~1
            transforms.Normalize((0.1307,), (0.3081,))
])

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transform,
                         download=True)

batch_size = 100
test_dataloader = DataLoader(mnist_test, batch_size= batch_size, shuffle=False)


""" inference accuracy in sw """
n_reps = 10   # Number of inference repetitions.

# inf_model = InfModel(model, "mnist")
# inf_model.sw_EvalModel(test_dataloader, n_reps)


""" inference accuracy in hw (simulator) """
# convert to aihwkit simulator
inf_model = InfModel(model, "mnist")
analog_model = inf_model.ConvertModel()  # convert s/w model to analog h/w model using aihwkit

# Inference
# t_inferences = [0.0, 3600.0, 86400.0, 1e7, 1e8, 1e9, 1e10, 1e12, 1e15]  # Times to perform infernece.
t_inferences = [0.0, 10.0, 100.0, 1000.0, 3600.0, 10000.0, 86400.0, 1e7, 1e8, 1e9, 1e10, 1e12, 1e15]
n_reps = 10   # Number of inference repetitions.
inf_model.hw_EvalModel(analog_model, test_dataloader, t_inferences, n_reps)


# ------------------------------------------------------------------
# measure run-time
sec = time.time()-start
times = str(datetime.timedelta(seconds=sec)) 
short = times.split(".")[0]   # until sec.
print(f"\nruntime : {short} sec\n")