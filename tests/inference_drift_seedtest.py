import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from Model.PyTorch_CIFAR10.cifar10_models.resnet import resnet18

# import customized files
from network import InfModel
import myModule

"""
inference tests over time

"""

# Setting
myModule.start_timer()
myModule.fix_seed()

dir_name = os.getcwd() + '/TestRun/'

name_list = [ 
            # 'vanilla-Resnet18',
            # 'Resnet18_p0.3',
            # 'Resnet18_p0.4',
            'Resnet18_p0.5',
            'Resnet18_p0.6',
            'Resnet18_p0.7',
            # 'Resnet18_p0.8',
            # 'Resnet18_p0.9',
               ]

# load the model
model_type = input("Input model type? (1: MLP / 2: Resnet18) : ")
imported_model = input("Input model type? (1: Pruned /2: Retrained / 3: Test) : ")

if imported_model == '1':
    model_name = 'local_pruned_model.pth'
elif imported_model == '2':
    model_name = 'FT_0.0001_50/best_model.pth'
elif imported_model == '3':
    model_name = 'FT_rev1.1_0.001_30/best_model.pth'
    
print(f'imported model : {model_name}')

# set test dataloader
if model_type == '1':
    datatype = "mnist"
elif model_type == '2':
    datatype = "cifar10"
    
_, testloader = myModule.set_dataloader(data_type=datatype)

# simulation setting
ideal_io = True
gdc_list = [True, False]
g_list = None  # default = None  // [0.1905, 25] 
noise_list = [0, 0]  # pgm, read noise scale respectively

def sim_iter(model_name, n_rep_sw: int, n_rep_hw: int) -> list :
    
    all_results = []
    
    # iteration
    for folder_name in name_list:
        print(f'\nfolder : {folder_name}')
        print(f'model : {model_name}')
        
        """ load the model """
        folder_path = dir_name + folder_name
        
        if 'vanilla' in folder_name:
            model = resnet18(pretrained=True)
            print('Vanilla model loaded')
        else:
            model = torch.load(f'{folder_path}/{model_name}', map_location='cpu')
        
        
        """ inference accuracy in sw """
        inf_model = InfModel(model, datatype)
        inf_model.sw_EvalModel(testloader, n_rep_sw)
        
        
        """ inference accuracy in hw (simulator) """ 
        rep_results = []  # Store results for each repetition
        
        # for rep in range(n_rep_hw):
        for rep in tqdm(range(n_rep_hw), desc='Inference Progress', 
                bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}'):
            
            # Set different seed for each repetition
            current_seed = 42 + rep
            
            inf_model = InfModel(model=model, mode=datatype, g_list=g_list, noise_list=noise_list)
            analog_model = inf_model.ConvertModel(gdc=gdc, ideal_io=ideal_io)
            
            # Single inference run with current seed
            t_inferences = [1,                         # 1 sec
                            60,                        # 1 min
                            100,
                            60 * 60,                   # 1 hour
                            24 * 60 * 60,              # 1 day
                            30 * 24 * 60 * 60,         # 1 month
                            12 * 30 * 24 * 60 * 60,    # 1 year
                            36 * 30 * 24 * 60 * 60,    # 3 year
                            1e9,
                            ]
            single_result = inf_model.hw_EvalModel_single(analog_model, testloader, t_inferences, seed=current_seed, n_reps=1)
            rep_results.append(single_result)
            
            myModule.clear_memory()
        
        # Calculate statistics across repetitions
        for t_idx in range(len(t_inferences)):
            t = t_inferences[t_idx]
            accuracies = [rep[t_idx][1] for rep in rep_results]  # Get accuracies for current time point
            
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            
            all_results.append([folder_name, t, mean_acc, std_acc])
            print(f"Time {t}s - Mean: {mean_acc:.2f}%, Std: {std_acc:.2f}%")


        myModule.clear_memory()
    return all_results

# simulation
n_rep_sw = 1   # Number of inference repetitions.
n_rep_hw = 30

for gdc in gdc_list:
    print(f'--- Ideal-IO:{ideal_io}, GDC:{gdc}, G range={g_list}, noise={noise_list} ---')
    all_results = sim_iter(model_name, n_rep_sw, n_rep_hw)
    
    df = pd.DataFrame(all_results, columns=["model", "Time (s)", "Mean Accuracy", "Std Accuracy"])
    df.to_excel(f"evaluation_results_gdc_{gdc}_seedtest.xlsx", index=False, engine='openpyxl')
    print(f"! Save the file: evaluation_results_gdc_{gdc}.xlsx\n")

# tracing ends
# tracelog.save_trace_results()

# temporary code---------------
model_name2 = 'FT_rev1.1_0.0001_30/best_model.pth'
print(model_name2)

for gdc in [True, False]:
    print(f'--- Ideal-IO:{ideal_io}, GDC:{gdc}, G range={g_list}, noise={noise_list} ---')
    all_results2 = sim_iter(model_name2, n_rep_sw, n_rep_hw)
    
    df2 = pd.DataFrame(all_results2, columns=["model", "Time (s)", "Mean Accuracy", "Std Accuracy"])
    df2.to_excel(f"evaluation_results_gdc_{gdc}_seedtest2.xlsx", index=False, engine='openpyxl')
    print(f"! Save the file 2 ! \n")
# -------------------------------

# # temporary code---------------
model_name3 = 'FT_rev1.1_1e-05_30/best_model.pth'
print(model_name3)

for gdc in gdc_list:
    print(f'--- Ideal-IO:{ideal_io}, GDC:{gdc}, G range={g_list}, noise={noise_list} ---')
    all_results3 = sim_iter(model_name3, n_rep_sw, n_rep_hw)
    
    df3 = pd.DataFrame(all_results3, columns=["model", "Time (s)", "Mean Accuracy", "Std Accuracy"])
    df3.to_excel(f"evaluation_results_gdc_{gdc}_seedtest3.xlsx", index=False, engine='openpyxl')
    print(f"! Save the file 3 ! \n")
# -------------------------------

 # temporary code---------------
model_name4 = 'FT_rev1.1_1e-06_30/best_model.pth'
print(model_name4)

for gdc in gdc_list:
    print(f'--- Ideal-IO:{ideal_io}, GDC:{gdc}, G range={g_list}, noise={noise_list} ---')
    all_results4 = sim_iter(model_name4, n_rep_sw, n_rep_hw)
    
    df4 = pd.DataFrame(all_results4, columns=["model", "Time (s)", "Mean Accuracy", "Std Accuracy"])
    df4.to_excel(f"evaluation_results_gdc_{gdc}_seedtest4.xlsx", index=False, engine='openpyxl')
    print(f"! Save the file 4 ! \n")
# -------------------------------

myModule.end_timer()