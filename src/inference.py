# -*- coding: utf-8 -*-

""" iteration for inference with different, fixed seed values """

import os
import sys
import numpy as np 
from tqdm import tqdm
from typing import Optional

import torch

# aihwkit related methods
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.presets import PCMPresetUnitCell

# custmized noise model
from aihwkit_test.customized_noise_pcm import TestNoiseModel

import myModule
from train import TrainModel

# get the parent directory and import the model
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from model.PyTorch_CIFAR10.cifar10_models.resnet import resnet18


class InferenceModel(TrainModel):
    """ Class for inference model """
    
    def __init__(self, model, model_name:str, datatype="cifar10", n_rep_sw: int=1, n_rep_hw: int=30):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model_name = model_name
        self.datatype = datatype
        _, self.testloader = myModule.set_dataloader(data_type=datatype)
        self.n_rep_sw = n_rep_sw
        self.n_rep_hw = n_rep_hw
     
        
    def run(self, 
            g_list: Optional[list] = None,
            noise_list: Optional[list]=None,
            gdc: bool= True,
            ideal_io: bool= False,
            inp_res_bit=7, 
            inp_noise=0.0, 
            out_res_bit=9, 
            out_noise=0.06
            ) -> list:
        
        """ Run inference with different parameters """
        
        pass

        # input paramter 마다 for 문 돌아가게 작성
        # for gdc in gdc_list:
        #     for io_res in io_res_list:
        #         for noise in noise_list:
        #             for g in g_list:
        #                 self.sim_iter(g_list=g, noise_list=noise, gdc=gdc, ideal_io=ideal_io,
        #                               inp_res_bit=inp_res_bit, inp_noise=inp_noise,
        #                               out_res_bit=out_res_bit, out_noise=out_noise)


        
    def sim_iter(self,
                 g_list: Optional[list] = None,
                 noise_list: Optional[list]=None,
                 gdc: bool= True,
                 ideal_io: bool= False,
                 inp_res_bit=7, 
                 inp_noise=0.0, 
                 out_res_bit=9, 
                 out_noise=0.06
                 ) -> list :

        """ Run inference with software and hardware """
        
        all_results = []            
            
        # inference accuracy in software
        self.SWinference()
        
        # inference accuracy in hardware (simulator) 
        t_inferences, rep_results = self.HWinference(self.model, self.testloader, 
                                datatype=self.datatype,
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
        all_results = self.acc_over_time(t_inferences, rep_results, all_results)

        myModule.clear_memory()
            
        return all_results


    def HWinference(
        self,
        g_list: Optional[list] = None,
        noise_list: Optional[list]=None,
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
        for rep in tqdm(range(self.n_rep_hw), desc='Inference Progress', 
                bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}'):
            
            # Set different seed for each repetition
            current_seed = 42 + rep
            
            analog_model = self.ConvertModel(gdc=gdc, 
                                             ideal_io=ideal_io,
                                             g_list=g_list,
                                             noise_list=noise_list,
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
            single_result = self.HWinference_single(analog_model, t_inferences, seed=current_seed)
            rep_results.append(single_result)
            
            # myModule.clear_memory()
            
        return t_inferences, rep_results
    
    
    def acc_over_time(self, t_inferences, rep_results, all_results) -> list:
        """ Calculate statistics across repetitions"""

        for t_idx in range(len(t_inferences)):
            t = t_inferences[t_idx]
            accuracies = [rep[t_idx][1] for rep in rep_results]  # Get accuracies for each time point
            
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            
            all_results.append([self.model_name, t, mean_acc, std_acc])
            print(f"Time {t}s - Mean: {mean_acc:.2f}%, Std: {std_acc:.2f}%")
            
        # myModule.clear_memory()
        
        return all_results
            
    
    def SWinference(self):
        """ Evaluate the model in software """
        self.model.to(self.device)
        self.model.eval()

        inference_accuracy_values = torch.zeros(self.n_rep_sw)
        myModule.fix_seed()

        for i in range(self.n_rep_sw):
            _, test_accuracy = self.eval_fn(self.model, self.testloader)
            inference_accuracy_values[i] = test_accuracy
        
        mean_acc = inference_accuracy_values.mean().item()
        std_acc = inference_accuracy_values.std().item()
            
        print(
                f"Test set accuracy (%) in s/w: \t mean: {mean_acc :.6f}, \t std: {std_acc :.6f}"
            )
    
    def HWinference_single(self, analog_model, t_inferences, seed=42):
        """ Evaluate the model in hardware for a single inference """
        analog_model.to(self.device)
        analog_model.eval()
        
        inference_accuracy_values = torch.zeros((len(t_inferences), 1)) 
        results = []
                
        for t_id, t in enumerate(t_inferences):
            # fix seed for reproducibility when applying the gaussian noise
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            
            analog_model.drift_analog_weights(t)             
            # inference
            analog_model.drift_analog_weights(t)
            _, test_accuracy = self.eval_fn(analog_model, self.testloader)
            inference_accuracy_values[t_id, 0] = test_accuracy
                            
            results.append([t, test_accuracy])
            
        return results
    
    
    def get_eval_function(self):
        # Define a dictionary mapping modes to evaluation functions
        eval_function_map = {
            "mnist": self.eval_mnist_mlp,
            "cifar10": self.eval_cifar10
        }

        if self.datatype not in eval_function_map:
            raise ValueError(f"Invalid mode: {self.datatype}. Supported modes are: {list(eval_function_map.keys())}")

        return eval_function_map[self.datatype]
    
    
    def ConvertModel(
        self,
        gdc: bool, 
        ideal_io: bool = False,
        g_list: Optional[list] = None,
        noise_list: Optional[list]=None,
        inp_res_bit: float = 7, 
        inp_noise: float = 0.0,           
        out_res_bit: float = 9, 
        out_noise: float = 0.06,
        ): 
        
        # fix seed for reproducibility during mapping
        myModule.fix_seed(seed=42)  
        
        pcm_config = self.SetConfig(
            gdc=gdc, 
            ideal_io=ideal_io,
            g_min=g_list[0] if g_list is not None else None,
            g_max=g_list[1] if g_list is not None else None,
            prog_noise_scale=noise_list[0] if noise_list is not None else None,
            read_noise_scale=noise_list[1] if noise_list is not None else None,
            inp_res_bit=inp_res_bit,
            inp_noise=inp_noise,
            out_res_bit=out_res_bit,
            out_noise=out_noise
            )
        
        analog_model = convert_to_analog(self.model, pcm_config)
        
        return analog_model
    
    
    def SetConfig(
        self, 
        gdc: bool, 
        ideal_io: bool = False,
        g_max: Optional[float] = None,
        g_min: Optional[float] = None,
        prog_noise_scale: Optional[float] = None,
        read_noise_scale: Optional[float] = None,
        inp_res_bit: float = 7, 
        inp_noise: float = 0.0,           
        out_res_bit: float = 9, 
        out_noise: float = 0.06,
        ):       
        
        rpu_config = InferenceRPUConfig()
        rpu_config.device = PCMPresetUnitCell()      # paired PCM devices (Gp-Gm)
        rpu_config.mapping.weight_scaling_omega = 1.0  
        
        # customized noise model
        rpu_config.noise_model = TestNoiseModel(
            g_max=g_max, 
            g_min=g_min,
            prog_noise_scale=prog_noise_scale,
            read_noise_scale=read_noise_scale,
            )  
        
        # global drift compensation
        if gdc == True: pass
        elif gdc == False:
            rpu_config.drift_compensation = None   
            
        # IO parameter settings
        if ideal_io == True:
            rpu_config.forward.is_perfect=True   
             
        elif ideal_io == False: 
            # set parameters for non-ideal IO
            rpu_config.forward = IOParameters(
                is_perfect=False,

                # === DAC (Input side) ===
                inp_bound=1.0,                           # DAC input range: [-1, 1]
                inp_res= 1.0 / (2**inp_res_bit - 2),     # n-bit DAC quantization
                inp_noise= inp_noise,
                # inp_sto_round=False,          # enable stochastic rounding in DAC
                # inp_asymmetry=0.0,            # 1% asymmetry in pos/neg DAC signal

                # === ADC (Output side) ===
                out_bound=12.0,                         # ADC saturation limit (max current)
                out_res= 1.0 / (2**out_res_bit - 2),    # n-bit DAC quantization      
                out_noise= out_noise,         
                # out_noise_std=0.1,             # 10% std variation across outputs
                # out_sto_round=False,            # enable stochastic rounding in ADC
                # out_asymmetry=0.005,           # 0.5% asymmetry in negative pass output

                # === Bound & Noise management (recommended for analog) : As default setting ===
                # bound_management=BoundManagementType.ITERATIVE,
                # noise_management=NoiseManagementType.ABS_MAX,

                # === etc. non-ideality : As default setting===
                # w_noise=0.0,                   
                # w_noise_type=WeightNoiseType.NONE,
                # ir_drop=0.0,
                # out_nonlinearity=0.0,
                # r_series=0.0
                
                # from example (if needed)
                # out_res = -1.0  # Turn off (output) ADC discretization.
                # w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
                # w_noise = 0.02  # Short-term w-noise.       
            )

        return rpu_config
    