# -*- coding: utf-8 -*-

""" inference tests over time to evaluate the impact of resistance drift """

# import customized files
import module.myModule as myModule
from module.model_loader import ModelLoader 
from module.inference import AdaBSConfig, InferenceModel


# Setting
myModule.start_timer()
myModule.fix_seed()

# load the model
imported_model = input("Model type? (1: Pruned /2: FineTuned / 3: Test) : ")
model_dict = ModelLoader.load_models(imported_model)


# simulation setting
gdc_list = [True]
io_list = [False]
noise_list = [[0,0,0], [0,0,1], [0,1,1], [1,1,1]]     # [[0,0,2.5], [0,0,0.5], [0,0,1.5]]     
g_list = [[0.1, 25]]          # add more: [[0.1,25], [0.2,25]]
io_res_list = [[7, 7]]      # ex) [[7, 7]]
io_noise_list = [[0.0, 0.0]]       # ex) [[0.0, 0.0]]

adabs = AdaBSConfig(
    enable=True,
    num_batches=13,      # ResNet32/CIFAR10 default from AdaBS paper setup
    batch_size=200,
    momentum=None,       # None -> p = 0.015 ** (1 / n)
)

# Run simulation
inference_model = InferenceModel(
    n_rep_hw=30,
    mapping_method="naive",   # DCM, DCM_1yr, naive
    model_dict=model_dict,
    gdc_list=gdc_list,
    io_list=io_list,
    noise_list=noise_list,
    g_list=g_list,
    io_res_list=io_res_list,
    io_noise_list=io_noise_list,
    adabs=adabs,
    distortion_f=0,
    compensation_alpha='auto',   # auto, LRS
)
inference_model.run()

inference_model2 = InferenceModel(
    n_rep_hw=30,
    mapping_method="DCM",   # DCM, DCM_1yr, naive
    model_dict=model_dict,
    gdc_list=gdc_list,
    io_list=io_list,
    noise_list=noise_list,
    g_list=g_list,
    io_res_list=io_res_list,
    io_noise_list=io_noise_list,
    adabs=adabs,
    distortion_f=0,
    compensation_alpha='auto',   # auto, LRS
)
inference_model2.run()


myModule.end_timer()
