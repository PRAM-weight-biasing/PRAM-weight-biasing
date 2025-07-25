# -*- coding: utf-8 -*-

""" inference tests over time to evaluate the impact of resistance drift """

# import customized files
import module.myModule as myModule
from module.model_loader import ModelLoader 
from module.inference import InferenceModel


# Setting
myModule.start_timer()
myModule.fix_seed()

# load the model
imported_model = input("Model type? (1: Pruned /2: FineTuned / 3: Test) : ")
model_dict = ModelLoader.load_models(imported_model)


# simulation setting
io_res_list = [[6,7], [6,8], [6,9], [7,7], [7,8], [7,9], [8,7], [8,8], [8,9]]  # [[7,9]]
io_noise_list = [[0.0, 0.0]]  

# Run simulation
inference_model = InferenceModel(
    n_rep_hw=30, 
    mapping_method="naive",
    model_dict=model_dict,
    gdc_list= [True],
    io_list= [False],
    noise_list= [[0, 0, 1]],           
    g_list= [[0.1, 25]],                 
    io_res_list=io_res_list,      
    io_noise_list=io_noise_list, 
    distortion_f= 0.0,
    compensation_alpha = 'auto',
    )
inference_model.run()


myModule.end_timer()