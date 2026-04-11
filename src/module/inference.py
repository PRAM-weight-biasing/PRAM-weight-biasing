# -*- coding: utf-8 -*-

""" iteration for inference with different, fixed seed values """

import os
import sys
from dataclasses import dataclass
from itertools import product
import numpy as np
import pandas as pd 
from tqdm import tqdm
from typing import Optional

import torch
import torch.nn as nn

# aihwkit related methods
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.presets import PCMPresetUnitCell

# custmized noise model
from module.noise_pcm import TestNoiseModel, MappingNoiseModel

import module.myModule as myModule
from module.train import TrainModel


@dataclass(frozen=True)
class AdaBSConfig:
    enable: bool = False
    num_batches: int = 13
    batch_size: int = 200
    momentum: Optional[float] = None


@dataclass(frozen=True)
class InferenceCondition:
    gdc: bool = True
    ideal_io: bool = False
    noise: Optional[list] = None
    g_list: Optional[list] = None
    io_res_bits: Optional[list] = None
    io_noise: Optional[list] = None

    @property
    def inp_res_bit(self):
        return None if self.io_res_bits is None else self.io_res_bits[0]

    @property
    def out_res_bit(self):
        return None if self.io_res_bits is None else self.io_res_bits[1]

    @property
    def inp_noise(self):
        return None if self.io_noise is None else self.io_noise[0]

    @property
    def out_noise(self):
        return None if self.io_noise is None else self.io_noise[1]


class InferenceModel(TrainModel):
    """ Class for inference model """
    
    def __init__(
        self, 
        model_dict: dict,
        datatype="cifar10", 
        n_rep_sw: int=1, 
        n_rep_hw: int=30, 
        mapping_method="naive",
        conditions: Optional[list] = None,
        adabs: Optional[AdaBSConfig] = None,
        gdc_list: Optional[list]=None,
        io_list: Optional[list]=None,
        noise_list: Optional[list]=None,         
        g_list: Optional[list] =None,             
        io_res_list: Optional[list] =None,       
        io_noise_list: Optional[list] = None,   
        distortion_f: Optional[float] = None,
        compensation_alpha: Optional[str] = 'auto',
        adabs_enable: bool = False,
        adabs_num_batches: int = 13,
        adabs_batch_size: int = 200,
        adabs_momentum: Optional[float] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_name = None
        self.model_dict = model_dict
        self.datatype = datatype
        self.trainloader, self.testloader = myModule.set_dataloader(data_type=datatype)
        self.n_rep_sw = n_rep_sw
        self.n_rep_hw = n_rep_hw
        self.mapping_method = mapping_method
        self.distortion_f = distortion_f
        self.compensation_alpha = compensation_alpha
        self.conditions = conditions or self.build_conditions(
            gdc_list=gdc_list,
            io_list=io_list,
            noise_list=noise_list,
            g_list=g_list,
            io_res_list=io_res_list,
            io_noise_list=io_noise_list,
        )
        self.adabs = adabs or AdaBSConfig(
            enable=adabs_enable,
            num_batches=adabs_num_batches,
            batch_size=adabs_batch_size,
            momentum=adabs_momentum,
        )
        self.adabs_loader = None
     
    @staticmethod
    def _normalize_grid(value, treat_list_as_single: bool = False) -> list:
        """Normalize an option value to a list of candidates.

        Examples:
        - value=None -> [None]
        - value=True -> [True]
        - value=[True, False] -> [True, False]
        - value=[0, 0, 1] with treat_list_as_single=True -> [[0, 0, 1]]
        - value=[[0, 0, 1], [0, 0, 2]] with treat_list_as_single=True -> [[...], [...]]
        """

        if value is None:
            return [None]

        if isinstance(value, tuple):
            value = list(value)

        if not isinstance(value, list):
            return [value]

        if len(value) == 0:
            return [None]

        if not treat_list_as_single:
            return value

        first_item = value[0]
        if len(value) == 1 and first_item is None:
            return [None]

        # Already a list of variants (eg [[...], [...]] or [None, [...]])
        if all((item is None) or isinstance(item, (list, tuple)) for item in value):
            return [list(item) if isinstance(item, tuple) else item for item in value]

        if isinstance(first_item, tuple):
            return [list(item) if isinstance(item, tuple) else item for item in value]
        if isinstance(first_item, list):
            return value

        return [value]

    @staticmethod
    def build_conditions(
        gdc_list: Optional[list] = None,
        io_list: Optional[list] = None,
        noise_list: Optional[list] = None,
        g_list: Optional[list] = None,
        io_res_list: Optional[list] = None,
        io_noise_list: Optional[list] = None,
    ) -> list:
        """Create a flat list of inference conditions from option grids."""

        normalized_gdc = InferenceModel._normalize_grid(
            [True] if gdc_list is None else gdc_list,
            treat_list_as_single=False,
        )
        normalized_io = InferenceModel._normalize_grid(
            [False] if io_list is None else io_list,
            treat_list_as_single=False,
        )
        normalized_noise = InferenceModel._normalize_grid(
            noise_list,
            treat_list_as_single=True,
        )
        normalized_g = InferenceModel._normalize_grid(
            g_list,
            treat_list_as_single=True,
        )
        normalized_io_res = InferenceModel._normalize_grid(
            io_res_list,
            treat_list_as_single=True,
        )
        normalized_io_noise = InferenceModel._normalize_grid(
            io_noise_list,
            treat_list_as_single=True,
        )

        return [
            InferenceCondition(
                gdc=gdc,
                ideal_io=ideal_io,
                noise=noise,
                g_list=g_range,
                io_res_bits=io_res_bits,
                io_noise=io_noise,
            )
            for ideal_io, gdc, noise, g_range, io_res_bits, io_noise in product(
                normalized_io,
                normalized_gdc,
                normalized_noise,
                normalized_g,
                normalized_io_res,
                normalized_io_noise,
            )
        ]

    def run(self) -> None:
        """ Run inference with different parameters """

        for condition in self.conditions:
            print(self.describe_condition(condition))
            self.run_one_condition(condition)


    def describe_condition(self, condition: InferenceCondition) -> str:
        msg = (
            f"\nRunning inference with mapping={self.mapping_method}"
            f" | gdc={condition.gdc}"
            f" | ideal_io={condition.ideal_io}"
            f" | noise={condition.noise}"
        )
        if condition.g_list is not None:
            msg += f"| g_list={condition.g_list}"
        if condition.io_res_bits is not None:
            msg += f"| io_res_bit={condition.io_res_bits}"
        if condition.io_noise is not None:
            msg += f"| io_noise={condition.io_noise}"
        if self.distortion_f is not None:
            msg += f"| distortion_f={self.distortion_f}"
        if self.compensation_alpha is not None:
            msg += f"| compensation={self.compensation_alpha}"
        if self.adabs.enable:
            msg += (
                f"| AdaBS=True"
                f"| adabs_num_batches={self.adabs.num_batches}"
                f"| adabs_batch_size={self.adabs.batch_size}"
            )
        return msg


    def run_one_condition(self, condition: InferenceCondition):
        
        # model loop
        all_results = []

        for model_name, model in self.model_dict.items():
            print(f"\n[Model: {model_name}]")

            self.model = model
            self.model_name = model_name

            results = self.sim_iter(condition)
            all_results.extend(results)

        # Save results
        io_res_tag = f"_io-{condition.io_res_bits}"
        adabs_tag = f"_adabs-{self.adabs.enable}"
        distortion_tag = ""
        if self.distortion_f is not None:
            distortion_tag = f"_distortion-{self.distortion_f:.1f}"
        filename = (
            f"../results/results_{self.mapping_method}"
            f"_gdc-{condition.gdc}"
            f"_io-{condition.ideal_io}"
            f"_noise-{condition.noise}"
            f"_{io_res_tag}{distortion_tag}{adabs_tag}.xlsx"
        )
        df = pd.DataFrame(all_results, columns=["model", "Time (s)", "Mean Accuracy", "Std Accuracy"])
        df.to_excel(filename, index=False, engine='openpyxl')
        print(f"Saved: {filename}")

        
    def sim_iter(self, condition: InferenceCondition) -> list :

        """ Run inference with software and hardware """
        
        results = []            
            
        # inference accuracy in software
        self.SWinference()
        
        # inference accuracy in hardware (simulator) 
        t_inferences, rep_results = self.HWinference(condition)
    
        # Calculate statistics across repetitions
        results = self.acc_over_time(t_inferences, rep_results, results)

        myModule.clear_memory()
            
        return results

    def HWinference(self, condition: InferenceCondition) -> list:
        
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
            
            analog_model = self.ConvertModel(condition)              
        
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
                nu_drift = 0.024282    # nu_min for applying DCM 
                manual_alpha = tau**nu_drift
                
                if self.compensation_alpha == 'auto': 
                    if self.mapping_method != "naive" : 
                        for tile in analog_model.analog_tiles():
                            tile.alpha = torch.tensor(manual_alpha, device=tile.alpha.device)
                        # print(f"[After override]  t={t}, manual alpha = {tile.alpha.item():.4f}")

                elif self.compensation_alpha == 'LRS':
                    # Use the user-defined alpha value 
                    for tile in analog_model.analog_tiles():
                        tile.alpha = torch.tensor(manual_alpha, device=tile.alpha.device)
                        # print(f"[After override]  t={t}, manual alpha = {tile.alpha.item():.4f}")
                        
                """ -------------- end ------------------------"""

                if self.adabs.enable:
                    self.apply_adabs(analog_model, current_seed + t_id)
                            
                                
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


    def get_adabs_loader(self):
        """Build a stable calibration loader for AdaBS.

        AdaBS is currently implemented for ResNet18 on CIFAR-10.
        The loader uses the CIFAR-10 training split without augmentation.
        """

        if self.datatype != "cifar10":
            raise ValueError("AdaBS is currently implemented only for CIFAR-10.")

        if self.adabs_loader is None:
            self.adabs_loader = myModule.set_cifar10_train_eval_loader(
                batch_size=self.adabs.batch_size,
                seed=42,
                shuffle=True,
            )

        return self.adabs_loader


    def get_adabs_momentum(self) -> float:
        if self.adabs.momentum is not None:
            return 1.0 - float(self.adabs.momentum)

        # PyTorch BN update rule:
        # running = (1 - momentum) * running + momentum * batch_stat
        # Paper rule:
        # running = p * running + (1 - p) * batch_stat
        # Therefore PyTorch momentum should be (1 - p)
        
        p = float(0.015 ** (1.0 / self.adabs.num_batches))
        torch_momentum = 1.0 - p
    
        return torch_momentum
    
    
    def _debug_bn_running_stats(self,model, tag=""):
        print(f"\n[DEBUG][{tag}] BN running stats")
        bn_idx = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                rm = module.running_mean.detach().float().cpu()
                rv = module.running_var.detach().float().cpu()
                print(
                    f"  BN{bn_idx:02d} {name}: "
                    f"mean_abs={rm.abs().mean().item():.6e}, "
                    f"mean_min={rm.min().item():.6e}, "
                    f"mean_max={rm.max().item():.6e}, "
                    f"var_mean={rv.mean().item():.6e}, "
                    f"var_min={rv.min().item():.6e}, "
                    f"var_max={rv.max().item():.6e}"
                )
                bn_idx += 1

    def apply_adabs(self, model, seed: int) -> None:
        """Recompute BN running stats using AdaBS calibration batches."""

        if self.datatype != "cifar10":
            raise ValueError("AdaBS is currently implemented only for CIFAR-10.")

        bn_layers = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        if not bn_layers:
            return
        
        
        momentum = self.get_adabs_momentum()
        loader = self.get_adabs_loader()

        myModule.fix_seed(seed)

        # Keep all non-BN modules in eval mode while BN layers update running stats.
        model.eval()
        
        # self._debug_bn_running_stats(model, tag="before_reset")
        
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train()
                module.momentum = momentum
                # module.reset_running_stats()
            else:
                module.eval()
        
        # self._debug_bn_running_stats(model, tag="after_reset")

        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(loader):
                if batch_idx >= self.adabs.num_batches:
                    break

                images = images.to(self.device)
                model(images)
                
        # self._debug_bn_running_stats(model, tag="after_adabs")

        model.eval()
        # print(
        #     f"[AdaBS] Recalibrated {len(bn_layers)} BN layers "
        #     f"with n={self.adabs.num_batches}, m={self.adabs.batch_size}, p={momentum:.6f}"
        # )
    
    
    def ConvertModel(self, condition: InferenceCondition): 
        
        # fix seed for reproducibility during mapping
        myModule.fix_seed(seed=42)  
        
        config_args = dict(
            gdc=condition.gdc,
            ideal_io=condition.ideal_io,
            g_min=condition.g_list[0] if condition.g_list is not None else None,
            g_max=condition.g_list[1] if condition.g_list is not None else None,
            prog_noise_scale=condition.noise[0] if condition.noise is not None else None,
            read_noise_scale=condition.noise[1] if condition.noise is not None else None,
            drift_noise_scale=condition.noise[2] if condition.noise is not None else None,
            inp_res_bit=condition.inp_res_bit,
            inp_noise=condition.inp_noise,
            out_res_bit=condition.out_res_bit,
            out_noise=condition.out_noise,
        )
             
        pcm_config = self._SetConfig(**config_args)
        analog_model = convert_to_analog(self.model, pcm_config)
        
        return analog_model
    
    # 260409
    def _SetConfig(
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
        
        # Define the conductance converter for DCM if needed
        if 'DCM' in self.mapping_method :
            from module.g_converter import MappedConductanceConverter
        
            if self.mapping_method == "DCM":
                g_conv = MappedConductanceConverter(
                    g_max=g_max,
                    g_min=g_min,
                    distortion_f=self.distortion_f,
                    profile="1month",
                )
            elif self.mapping_method == "DCM_1yr":
                g_conv = MappedConductanceConverter(
                    g_max=g_max,
                    g_min=g_min,
                    distortion_f=self.distortion_f,
                    profile="1year",
                )
            else:
                raise ValueError(f"Unsupported mapping_method: {self.mapping_method}")
            
        else:
            g_conv = None
        
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
            g_converter=g_conv, 
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
       
    
    def convert_all_models(self) -> dict:
        analog_models = {}
        default_condition = self.conditions[0] if self.conditions else InferenceCondition()
        for name, mdl in self.model_dict.items():
            self.model_name = name
            self.model = mdl
            analog_models[name] = self.ConvertModel(default_condition) 
            
        return analog_models
