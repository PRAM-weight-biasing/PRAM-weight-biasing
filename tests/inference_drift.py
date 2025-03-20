import os
import torch
import pandas as pd

from Model.PyTorch_CIFAR10.cifar10_models.resnet import resnet18
# from Model.PyTorch_CIFAR10.cifar10_models.vgg import vgg16_bn

# import customized files
from network import InfModel
import myModule

"""
inference tests over time

"""

# Setting
myModule.start_timer()
# myModule.fix_seed()

# trace the imported files in aihwkit folder
# tracelog = myModule.trace()
# tracelog.start_trace()

dir_name = os.getcwd() + '/TestRun/'
# dir_name = os.getcwd() + '/Model/'

# name_list = ["vanilla-MLP"]
name_list = [ 
             'vanilla-Resnet18',
             'Test_2024-10-28_15-15_Resnet18_p0.3',
             'Test_2024-10-28_15-22_Resnet18_p0.4',
             'Test_2024-10-28_15-26_Resnet18_p0.5',
             'Test_2024-10-28_15-27_Resnet18_p0.6',
             'Test_2024-10-28_15-32_Resnet18_p0.7',
               ]

# load the model
model_type = input("Input model type? (1: MLP / 2: Resnet18) : ")
imported_model = input("Input model type? (1: Pruned /2: Retrained) : ")

if imported_model == '1':
    model_name = 'local_pruned_model.pth'
elif imported_model == '2':
    model_name = 'FineTuning/best_model.pth'
    
print(f'imported model : {model_name}')

# set test dataloader
if model_type == '1':
    datatype = "mnist"
elif model_type == '2':
    datatype = "cifar10"
    
_, testloader = myModule.set_dataloader(data_type=datatype)

# simulation setting
ideal_io = True
gdc_list = [False, True]
g_list = None  # defau0lt = None  // [0.1905, 25] 
noise_list = [0, 0]  # pgm, read noise scale respectively

def sim_iter(n_rep_sw: int, n_rep_hw: int) -> list :
    
    all_results = []
    # iteration
    for folder_name in name_list:
        print(f'\nfolder : {folder_name}')
        
        """ load the model """
        folder_path = dir_name + folder_name
        if 'vanilla' in folder_name:
            model = resnet18(pretrained=True)
        else:
            model = torch.load(f'{folder_path}/{model_name}')
        

        """ inference accuracy in sw """
        inf_model = InfModel(model, datatype)
        inf_model.sw_EvalModel(testloader, n_rep_sw)


        """ inference accuracy in hw (simulator) """ 
        inf_model = InfModel(model=model, mode=datatype, g_list=g_list, noise_list=noise_list)
        analog_model = inf_model.ConvertModel(gdc=gdc, ideal_io=ideal_io)  

        # Inference
        t_inferences = [
                        1,                         # 1 sec
                        60,                        # 1 min
                        100,
                        60 * 60,                   # 1 hour
                        24 * 60 * 60,              # 1 day
                        30 * 24 * 60 * 60,         # 1 month
                        12 * 30 * 24 * 60 * 60,    # 1 year
                        36 * 30 * 24 * 60 * 60,    # 3 year
                        1e9,
                        ]
        # [0.0, 10.0, 100.0, 1000.0, 3600.0, 10000.0, 86400.0, 1e6, 1e7, 1e8, 1e9]
        results = inf_model.hw_EvalModel(analog_model, testloader, t_inferences, n_rep_hw)
        
        for row in results:
            all_results.append([folder_name] + row)

        myModule.clear_memory()
    return all_results

# simulation
n_rep_sw = 1   # Number of inference repetitions.
n_rep_hw = 30  

for gdc in gdc_list:
    print(f'\n--- Ideal-IO:{ideal_io}, GDC:{gdc}, G range={g_list}, noise={noise_list} ---')
    all_results = sim_iter(n_rep_sw, n_rep_hw)
    
    df = pd.DataFrame(all_results, columns=["model", "Time (s)", "Mean Accuracy", "Std Accuracy"])
    df.to_excel(f"evaluation_results_gdc_{gdc}.xlsx", index=False, engine='openpyxl')
    print(f"! Save the file: evaluation_results_gdc_{gdc}.xlsx\n")

# tracing ends
# tracelog.save_trace_results()

myModule.end_timer()