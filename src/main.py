# -*- coding: utf-8 -*-

""" inference tests over time to evaluate the impact of resistance drift """

# import customized files
import myModule
from model_loader import ModelLoader 
from inference import InferenceModel


# Setting
myModule.start_timer()
myModule.fix_seed()

# load the model
imported_model = input("Model type? (1: Pruned /2: FineTuned / 3: Test) : ")
model_dict = ModelLoader.load_models(imported_model)


# simulation setting
io_res_list = [[7,9]],  #[[6,7], [6,8], [6,9], [7,7], [7,8], [7,9], [8,7], [8,8],[8,9]]  # inp_res, out_res
io_noise_list = [[0.0, 0.0]]   # inp_noise, out_noise


# Run simulation
inference_model = InferenceModel(n_rep_hw=30)
inference_model.run(
    model_dict=model_dict,
    gdc_list= [True, False],
    io_list= [False],
    noise_list= [0, 0],           
    g_list= None,                 
    io_res_list=io_res_list,      
    io_noise_list=io_noise_list, 
    )