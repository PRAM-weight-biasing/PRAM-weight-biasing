# -*- coding: utf-8 -*-

""" iteration for inference with different, fixed seed values """

import os
import sys
import numpy as np 
import torch
from tqdm import tqdm

import myModule

# get the parent directory and import the model
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from model.PyTorch_CIFAR10.cifar10_models.resnet import resnet18



def sim_iter(model_name, 
             testloader,
             n_rep_sw: int=1, 
             n_rep_hw: int=30, 
             g_list= None,
             noise_list= None,
             gdc: bool= True,
             ideal_io: bool= False,
             inp_res_bit=7, 
             inp_noise=0.0, 
             out_res_bit=9, 
             out_noise=0.06
             ) -> list :

    all_results = []
    
    # iteration
    for folder_name in name_list:
        print(f'\nfolder : {folder_name}')

        """ load the model """
        folder_path = dir_name + folder_name
        
        if 'vanilla' in folder_name:
            model = resnet18(pretrained=True)
            print('Vanilla model loaded')
        else:
            model = torch.load(f'{folder_path}/{model_name}', map_location='cpu')
            print(f'model : {model_name}')
        
        
        """ inference accuracy in sw """
        inf_model = InfModel(model, datatype="cifar10")
        inf_model.sw_EvalModel(testloader, n_rep_sw)
        
        """ inference accuracy in hw (simulator) """ 
        t_inferences, rep_results = HWinference(model, testloader, 
                                  n_rep_hw, 
                                  datatype="cifar10",
                                  g_list=g_list, 
                                  noise_list=noise_list,
                                  gdc=gdc, 
                                  ideal_io=ideal_io,
                                  inp_res_bit=inp_res_bit,
                                  inp_noise=inp_noise,
                                  out_res_bit=out_res_bit,
                                  out_noise=out_noise,
                                  )
 
        # Calculate statistics across repetitions
        for t_idx in range(len(t_inferences)):
            t = t_inferences[t_idx]
            accuracies = [rep[t_idx][1] for rep in rep_results]  # Get accuracies for each time point
            
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            
            all_results.append([folder_name, t, mean_acc, std_acc])
            print(f"Time {t}s - Mean: {mean_acc:.2f}%, Std: {std_acc:.2f}%")

        myModule.clear_memory()
        
    return all_results


def HWinference(
    model,
    testloader,
    n_rep_hw: int=30,
    datatype='cifar10',
    g_list= None,
    noise_list= None,
    gdc: bool= True,
    ideal_io: bool= False,
    inp_res_bit: float= 7,
    inp_noise: float= 0.0,           
    out_res_bit: float= 9,
    out_noise: float= 0.06,
    ) -> list:
    
    """ inference accuracy in hw (simulator) """ 
    rep_results = []  # Store results for each repetition
    
    # for rep in range(n_rep_hw):
    for rep in tqdm(range(n_rep_hw), desc='Inference Progress', 
            bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}'):
        
        # Set different seed for each repetition
        current_seed = 42 + rep
        
        inf_model = InfModel(model=model, mode=datatype, g_list=g_list, noise_list=noise_list)
        analog_model = inf_model.ConvertModel_io(gdc=gdc, ideal_io=ideal_io,
                                                    inp_res_bit=inp_res_bit,
                                                    inp_noise=inp_noise,
                                                    out_res_bit=out_res_bit,
                                                    out_noise=out_noise,
                                                    )
                                                
        
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
        
    return t_inferences, rep_results