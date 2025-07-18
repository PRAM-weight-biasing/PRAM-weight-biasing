"""
from aihwkit.inference.noise.pcm
"""
# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=too-many-instance-attributes

"""Phenomenological noise models for PCM devices for inference."""

from copy import deepcopy
from typing import List, Optional

from numpy import log as numpy_log
from numpy import sqrt, exp
from torch import abs as torch_abs
from torch import clamp, log, randn_like, Tensor
from torch.autograd import no_grad

from aihwkit.inference.noise.base import BaseNoiseModel
from aihwkit.inference.converter.base import BaseConductanceConverter
from aihwkit.inference.converter.conductance import SinglePairConductanceConverter



_ZERO_CLIP = 1e-7


class TestNoiseModel(BaseNoiseModel):
    r"""Noise model that was fitted and characterized on real PCM devices.

    Expected weight noise at assumed time of inference with expected
    programming noise at 0.

    The statistical noise model is based on measured PCM devices. See
    also `Nandakumar et al. ICECS (2019)`_

    Args:
        prog_coeff: Programming polynomial coeffs in
            :math:`\sum_i c_i \left(\frac{g_t}{g_\max}\right)^i`
        g_converter: instantiated class of the conductance converter
            (defaults to single pair)
        g_max: In :math:`\mu S`, the maximal conductance, ie the value
            the absolute max of the weights will be mapped to.
        t_read: Parameter of the 1/f fit (in seconds).
        t_0: Parameter of the drift fit (first reading time).

            Note:
                The ``t_inference`` is relative to this time `t0`
                e.g. t_inference counts from the completion of the programming
                of a device.
        prog_noise_scale: Scale for the programming noise.
        read_noise_scale: Scale for the read and accumulated noise.
        drift_scale: Scale for the  drift coefficient.
        prog_coeff_g_max_reference: reference :math:`g_\max` value
            when fitting the coefficients, since the result of the
            polynomial fit is given in uS. If
            ``prog_coeff_g_max_reference`` is not given and
            `prog_coeffs` are given explicitly, it will be set to
            ``g_max`` of the conductance converter.

    .. _`Nandakumar et al. ICECS (2019)`: https://ieeexplore.ieee.org/abstract/document/8964852

    """

    def __init__(
        self,
        prog_coeff: Optional[List[float]] = None,
        g_converter: Optional[BaseConductanceConverter] = None,
        g_max: Optional[float] = None,
        g_min: Optional[float] = None,
        t_read: float = 250.0e-9,
        t_0: float = 20.0,
        prog_noise_scale: float = 1.0,
        read_noise_scale: float = 1.0,
        drift_scale: float = 1.0,
        prog_coeff_g_max_reference: Optional[float] = None,
        drift_noise_scale: float = 1.0,
    ):
        g_converter = deepcopy(g_converter) or SinglePairConductanceConverter(g_max=g_max, g_min=g_min)
        super().__init__(g_converter)

        self.g_max = getattr(self.g_converter, "g_max", g_max)
        if self.g_max is None:
            raise ValueError("g_max cannot be established from g_converter")
        
        self.g_min = getattr(self.g_converter, "g_min", g_min) 
        if self.g_min is None:
            
            raise ValueError("g_min cannot be established from g_converter")

        if prog_coeff_g_max_reference is None:
            self.prog_coeff_g_max_reference = self.g_max

        if prog_coeff is None:
            # standard g_max are defined in respect to 25.0 uS. Need to
            # adjust for that in case g_max is not equal to 25.0 uS
            self.prog_coeff = [0.26348, 1.9650, -1.1731]
            self.prog_coeff_g_max_reference = 25.0
        else:
            self.prog_coeff = prog_coeff

        self.t_0 = t_0
        self.t_read = t_read
        self.prog_noise_scale = prog_noise_scale
        self.read_noise_scale = read_noise_scale
        self.drift_scale = drift_scale
        self.drift_noise_scale = drift_noise_scale

    @no_grad()
    def apply_programming_noise_to_conductance(self, g_target: Tensor) -> Tensor:
        """Apply programming noise to a target conductance Tensor.

        Programming noise with additive Gaussian noise with
        conductance dependency of the variance given by a 2-degree
        polynomial.
        """
        mat = 1
        sig_prog = self.prog_coeff[0]
        for coeff in self.prog_coeff[1:]:
            mat *= g_target / self.g_max
            sig_prog += mat * coeff

        sig_prog *= self.g_max / self.prog_coeff_g_max_reference  # type: ignore
        g_prog = g_target + self.prog_noise_scale * sig_prog * randn_like(g_target)
        g_prog.clamp_(min=0.0)  # no negative conductances allowed
        
        # Debug: 첫 번째 값만 1번만 출력
        # if not hasattr(self, "_print_once_program"):
        #     print("[DEBUG] First g_prog value:", g_prog.view(-1)[0].item())
        #     self._print_once_program = True

        return g_prog

    @no_grad()
    def generate_drift_coefficients(self, g_target: Tensor) -> Tensor:
        """Return drift coefficients ``nu`` based on PCM measurements."""
        # g_relative = clamp(torch_abs(g_target / self.g_max), min=_ZERO_CLIP)  
        g_relative = clamp(torch_abs((g_target - self.g_min) / (self.g_max-self.g_min)), min=_ZERO_CLIP)  # revision 
        
        # import torch
        # print(f"[DEBUG] torch.initial seed : {torch.initial_seed()}")
        
        # gt should be normalized wrt g_max
        """ mu_drift """
        mu_orig = (-0.0155 * log(g_relative) + 0.0244).clamp(min=0.049, max=0.1)
        mu_log = (-0.0155 * log(g_relative) + 0.0244).clamp(min=0.01) 
        mu_log_rev = (-0.0155 * log(g_relative+0.00762) + 0.0244)    # maximum value=0.1
        mu_linear = (-0.05 * g_relative + 0.1).clamp(min=_ZERO_CLIP) 
        mu_const_010 = 0.1
        mu_const_005 = 0.05
        mu_const_001 = 0.01
        mu_zero = 0
        mu_gst225 = -0.1841* (g_relative**3) + 0.4204* (g_relative**2) - 0.3134 * g_relative + 0.08465
        mu_msr = 0.0513*exp(-4.9751*g_relative**0.5803) + 0.0069
        
        mu_test1 = (-0.0155 * log((g_relative**0.5)+0.00762) + 0.0244)
        mu_test2 = (-0.0155 * log((g_relative**1.5)+0.00762) + 0.0244)
        mu_test3 = (-0.0757 * g_relative + 0.1)
        
        mu_test4 = (-0.0155 * log((g_relative*2)+0.00762) + 0.0244).clamp(min=0.0244)
        mu_test5 = (-0.0155 * log((g_relative*5)+0.00762) + 0.0244).clamp(min=0.0244)
        
        mu_drift = mu_log_rev  # final
        
        
        """ sig_drift """
        sig_orig = (-0.0125 * log(g_relative) - 0.0059).clamp(min=0.008, max=0.045)
        sig_const = 0.008
        sig_zero = 0
        
        sig_drift = sig_orig * self.drift_noise_scale  # final
        
        """ final nu """
        nu_drift = torch_abs(mu_drift + sig_drift * randn_like(g_relative)).clamp(min=0.0)
        
        # debugging
        # if not hasattr(self, "_print_once_drift"):
        #     print("[DEBUG] First nu_drift value:", nu_drift.view(-1)[0].item())
        #     self._print_once_drift = True

        return nu_drift * self.drift_scale

    @no_grad()
    def apply_drift_noise_to_conductance(
        self, g_prog: Tensor, drift_noise_param: Tensor, t_inference: float
    ) -> Tensor:
        """Apply the noise and drift up to the assumed inference time
        point based on PCM measurements."""
        t = t_inference + self.t_0

        # drift
        if t > self.t_0:
            g_drift = g_prog * ((t / self.t_0) ** (-drift_noise_param))
        else:
            g_drift = g_prog

        # expected accumulated 1/f noise since start of programming at t=0
        if t > 0:
            q_s = (0.0088 / ((torch_abs(g_prog) / self.g_max) ** 0.65).clamp(min=1e-3)).clamp(
                max=0.2
            )
            sig_noise = q_s * sqrt(numpy_log((t + self.t_read) / (2 * self.t_read)))
            g_final = g_drift + torch_abs(g_drift) * self.read_noise_scale * sig_noise * randn_like(
                g_prog
            )
            
            # if not hasattr(self, "_print_once_read"):
            #     print("[DEBUG] First sig_noise value:", sig_noise.view(-1)[0].item())
            #     self._print_once_read = True
                
        else:
            g_final = g_prog

        return g_final.clamp(min=0.0)

class MappingNoiseModel(TestNoiseModel):
    """_summary_

    Args:
        TestNoiseModel (_type_): _description_
    """

    def __init__(
        self,
        g_converter: Optional[BaseConductanceConverter] = None,
        g_max: Optional[float] = None,
        g_min: Optional[float] = None,
        t_0: float = 20.0,
        prog_noise_scale: float = 1.0,
        read_noise_scale: float = 1.0,
        drift_scale: float = 1.0,
        drift_noise_scale: float = 1.0,
        ):
        
        super().__init__(
            g_converter=g_converter,
            g_max=g_max,
            g_min=g_min,
            t_0=t_0,
            prog_noise_scale=prog_noise_scale,
            read_noise_scale=read_noise_scale,
            drift_scale=drift_scale,
            drift_noise_scale=drift_noise_scale,
        )
        
    @no_grad()
    def generate_drift_coefficients(self, g_target: Tensor) -> Tensor:
        """Return drift coefficients ``nu`` based on PCM measurements."""
        # g_relative = clamp(torch_abs(g_target / self.g_max), min=_ZERO_CLIP)  
        g_relative = clamp(torch_abs((g_target-self.g_min) / (self.g_max-self.g_min)), min=_ZERO_CLIP)  # for [0,1] range 
        
        
        # gt should be normalized wrt g_max
        """ mu_drift """
        mu_orig = (-0.0155 * log(g_relative) + 0.0244).clamp(min=0.049, max=0.1)
        mu_log_rev = (-0.0155 * log(g_relative+0.00762) + 0.0244)    # maximum value=0.1
        
        mu_drift = mu_log_rev  # final
        
        
        """ sig_drift """
        sig_orig = (-0.0125 * log(g_relative) - 0.0059).clamp(min=0.008, max=0.045)
        
        sig_drift = sig_orig * self.drift_noise_scale  # final
        
        
        """ final nu """
        nu_drift = torch_abs(mu_drift + sig_drift * randn_like(g_relative)).clamp(min=0.0)
        

        return nu_drift * self.drift_scale

    