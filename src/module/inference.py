# -*- coding: utf-8 -*-

""" iteration for inference with different, fixed seed values """

import os
import sys
import numpy as np
import pandas as pd 
from tqdm import tqdm
from typing import Optional

import torch

# aihwkit related methods
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.presets import PCMPresetUnitCell

# custmized noise model
from module.noise_pcm import TestNoiseModel, MappingNoiseModel

import module.myModule as myModule
from module.train import TrainModel


class InferenceModel(TrainModel):
    """ Class for inference model """
    
    def __init__(
        self, 
        model_dict: dict,
        datatype="cifar10", 
        n_rep_sw: int=1, 
        n_rep_hw: int=30, 
        mapping_method="naive",
        gdc_list: Optional[list]=[True],
        io_list: Optional[list]=[False],
        noise_list: Optional[list]=None,         
        g_list: Optional[list] =None,             
        io_res_list: Optional[list] =None,       
        io_noise_list: Optional[list] = None,   
        distortion_f: Optional[float] = None,
        compensation_alpha: Optional[str] = 'auto',
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_name = None
        self.model_dict = model_dict
        self.datatype = datatype
        _, self.testloader = myModule.set_dataloader(data_type=datatype)
        self.n_rep_sw = n_rep_sw
        self.n_rep_hw = n_rep_hw
        self.mapping_method = mapping_method
        self.gdc_list = gdc_list
        self.io_list = io_list
        self.noise_list = noise_list            # [program, read, drift sigma noise scale]
        self.g_list = g_list                    # [gmin, gmax] 
        self.io_res_list = io_res_list          # [inp_res, out_res]
        self.io_noise_list = io_noise_list      # [inp_noise, out_noise]
        self.distortion_f = distortion_f
        self.compensation_alpha = compensation_alpha
     
        
    def run(self) -> None:     
        """ Run inference with different parameters """

        # for loop for every input parameter
        for io in self.io_list:
            for gdc in self.gdc_list:
                for noise in self.noise_list :
                    for g in self.g_list or [None] :
                        for io_res_bit in self.io_res_list or [None] :
                            for io_noise in self.io_noise_list or [None] :
                                # print message
                                msg = f"\nRunning inference with mapping={self.mapping_method} | gdc={gdc} | ideal_io={io} | noise={noise}"
                                if g is not None: msg += f"| g_list={g}"
                                if io_res_bit is not None: msg += f"| io_res_bit={io_res_bit}"
                                if io_noise is not None: msg += f"| io_noise={io_noise}"
                                if self.distortion_f is not None: msg += f"| distortion_f={self.distortion_f}"
                                if self.compensation_alpha is not None: msg += f"| compensation={self.compensation_alpha}"
                                print(msg)
                                
                                self.run_one_condition(gdc, io, noise, g, io_res_bit, io_noise)


    def run_one_condition(self, gdc, io, noise, g, io_res_bit, io_noise):
        
        # the arguments for 'sim_iter'
        kwargs = {
            "gdc": gdc,
            "ideal_io": io,
            "noise_list": noise,
        }
        if g is not None:
            kwargs["g_list"] = g

        if io_res_bit is not None:
            inp_res_bit, out_res_bit = io_res_bit
            kwargs.update({
                "inp_res_bit": inp_res_bit,
                "out_res_bit": out_res_bit,
            })

        if io_noise is not None:
            inp_noise, out_noise = io_noise
            kwargs.update({
                "inp_noise": inp_noise,
                "out_noise": out_noise,
            })

        # model loop
        all_results = []

        for model_name, model in self.model_dict.items():
            print(f"\n[Model: {model_name}]")

            self.model = model
            self.model_name = model_name

            results = self.sim_iter(**kwargs)
            all_results.extend(results)

        # Save results
        filename = f"../results/results_{self.mapping_method}_gdc-{gdc}_io-{io}_noise-{noise}_{io_res_bit}.xlsx"
        df = pd.DataFrame(all_results, columns=["model", "Time (s)", "Mean Accuracy", "Std Accuracy"])
        df.to_excel(filename, index=False, engine='openpyxl')
        print(f"Saved: {filename}")

        
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
        
        results = []            
            
        # inference accuracy in software
        self.SWinference()
        
        # inference accuracy in hardware (simulator) 
        t_inferences, rep_results = self.HWinference(
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
        results = self.acc_over_time(t_inferences, rep_results, results)

        myModule.clear_memory()
            
        return results

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
        
        # for rep in range(n_rep_hw):
        for rep in tqdm(range(self.n_rep_hw), desc='Inference Progress', 
                bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}'):
            
            # Set different seed for each repetition
            current_seed = 42 + rep
            myModule.fix_seed(current_seed)     
            
            analog_model = self.ConvertModel(
            gdc=gdc,
            ideal_io=ideal_io,
            g_list=g_list,
            noise_list=noise_list,
            inp_res_bit=inp_res_bit,
            inp_noise=inp_noise,
            out_res_bit=out_res_bit,
            out_noise=out_noise,
            )              
        
            analog_model.to(self.device)
            analog_model.eval() 
            
            # Inference with different time points
            results = []
            for t_id, t in enumerate(t_inferences):
                
                 # fix seed for reproducibility when applying the gaussian noise
                torch.manual_seed(current_seed)
                torch.cuda.manual_seed(current_seed)
                
                # inference
                analog_model.drift_analog_weights(t)
                
                """ Change drift compensation factor alpha ------------"""
                # 원래 자동 보정된 alpha 출력
                # for tile in analog_model.analog_tiles():
                    # print(f"[Before override] t={t}, auto alpha = {tile.alpha.item():.4f}")

                #  change the amplification(drift compensation) factor
                tau = 1 + t/20         # (t0+t)/t0
                nu_drift = 0.024282    # nu_max 
                manual_alpha = tau**nu_drift
                
                if self.compensation_alpha == 'auto': 
                    if self.mapping_method != "naive" : 
                        for tile in analog_model.analog_tiles():
                            tile.alpha = torch.tensor(manual_alpha, device=tile.alpha.device)
                        # print(f"[After override]  t={t}, manual alpha = {tile.alpha.item():.4f}")

                elif self.compensation_alpha == 'max':
                    # Use the user-defined alpha value 
                    for tile in analog_model.analog_tiles():
                        tile.alpha = torch.tensor(manual_alpha, device=tile.alpha.device)
                        # print(f"[After override]  t={t}, manual alpha = {tile.alpha.item():.4f}")
                        
                """ -------------- end ------------------------"""
                            
                                
                _, test_accuracy = self.get_eval_function()(analog_model, self.testloader)
                results.append([t, test_accuracy])
                # print(f'[DEBUG] t={t}, test_accuracy={test_accuracy:.4f}')
                                
            rep_results.append(results)
            
            myModule.clear_memory()
            
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
            _, test_accuracy = self.get_eval_function()(self.model, self.testloader)
            inference_accuracy_values[i] = test_accuracy
        
        mean_acc = inference_accuracy_values.mean().item()
        if self.n_rep_sw > 1:
            std_acc = inference_accuracy_values.std().item()
        else:  std_acc = 0.0
            
        print(
                f"Test set accuracy (%) in s/w: \t mean: {mean_acc :.6f}, \t std: {std_acc :.6f}"
            )
    
    
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
        
        config_args = dict(
            gdc=gdc,
            ideal_io=ideal_io,
            g_min=g_list[0] if g_list is not None else None,
            g_max=g_list[1] if g_list is not None else None,
            prog_noise_scale=noise_list[0] if noise_list is not None else None,
            read_noise_scale=noise_list[1] if noise_list is not None else None,
            drift_noise_scale=noise_list[2] if noise_list is not None else None,
            inp_res_bit=inp_res_bit,
            inp_noise=inp_noise,
            out_res_bit=out_res_bit,
            out_noise=out_noise
        )

        # Set the mapping methods       
        if self.mapping_method == "naive":
            pcm_config = self.SetConfig(**config_args)
        elif self.mapping_method == "myMapping":
            # for customized Gp-Gm mapping
            pcm_config = self.MappingSetConfig(**config_args)
        
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
        drift_noise_scale: Optional[float] = None,
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
            drift_noise_scale=drift_noise_scale,
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


    def MappingSetConfig(
        self, 
        gdc: bool, 
        ideal_io: bool = False,
        g_max: Optional[float] = None,
        g_min: Optional[float] = None,
        prog_noise_scale: Optional[float] = None,
        read_noise_scale: Optional[float] = None,
        drift_noise_scale: Optional[float] = None,
        inp_res_bit: float = 7, 
        inp_noise: float = 0.0,           
        out_res_bit: float = 9, 
        out_noise: float = 0.06,
        ):       
        
        from module.g_converter import MappedConductanceConverter
        
        rpu_config = InferenceRPUConfig()
        rpu_config.device = PCMPresetUnitCell()      # paired PCM devices (Gp-Gm)
        rpu_config.mapping.weight_scaling_omega = 1.0  
        
        # customized noise model
        rpu_config.noise_model = MappingNoiseModel(
            g_max=g_max, 
            g_min=g_min,
            prog_noise_scale=prog_noise_scale,
            read_noise_scale=read_noise_scale,
            drift_noise_scale=drift_noise_scale,
            g_converter=MappedConductanceConverter(g_max=g_max, g_min=g_min, distortion_f=self.distortion_f), 
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