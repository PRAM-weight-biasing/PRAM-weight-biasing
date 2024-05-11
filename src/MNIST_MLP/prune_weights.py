import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import numpy as np
import copy

"""=============================================================="""
class PruneModel():
    def __init__(self, model: nn.Sequential, save_dir: str):
        self.save_dir       = save_dir
        self.model_vanilla  = model
        self.model_pruned   = None


    def local_pruning(self, prune_pct: float) -> nn.Module:
        """prune (prune_pct) percent of weights from each layer

        Args:
            prune_pct (float): percentage of weights to be pruned from each layer

        Returns:
            nn.Module: pruned network
        """
        model_to_prune = copy.deepcopy(self.model_vanilla)
        # apply pruning for every layer
        for _, module in model_to_prune.named_modules():
            if isinstance(module, nn.Linear):
                # currently pruning with a fixed method
                prune.l1_unstructured(module, 'weight', amount=prune_pct)
                prune.remove(module, 'weight')
        # save pruned model
        torch.save(model_to_prune, f'{self.save_dir}/prune_local_{prune_pct*100: 2.0f}.pth')
        
        return model_to_prune

    def global_pruning(self, prune_pct: float) -> nn.Module:
        """prune (prune_pct) percent of weights globally

        Args:
            prune_pct (float): percentage of weights to be globally pruned

        Returns:
            nn.Module: pruned network
        """
        model_to_prune      = copy.deepcopy(self.model_vanilla)
        parameters_to_prune = (
            (model_to_prune.model.fc1, 'weights'),
            (model_to_prune.model.fc2, 'weights'),
            (model_to_prune.model.fc3, 'weights')
        )
        
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.l1_unstructured,
                                  amount=prune_pct)
        
        for _, module in model_to_prune.named_modules():
            if isinstance(module, nn.Linear):
                prune.remove(module, 'weight')
                
        torch.save(model_to_prune, f'{self.save_dir}/prune_global_{prune_pct*100: 2.0f}.pth')        
        
        return model_to_prune

    def calc_sparsity(self, pruned_model: nn.Module) -> tuple[float, list]:
        """calculate sparsity of network globally/locally

        Args:
            pruned_model (nn.Module): input network of interest

        Returns:
            tuple[float, list]: (global sparsity, layer-wise list of local sparsity)
        """
        total_params_num    = 0
        total_zeros_num     = 0
        sparsity_list       = []
        
        for _, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear):
                # total number of parameters in target layer
                layer_params_num    = module.weight.numel()
                # total number of zero weights in target layer 
                zero_params_num     = torch.sum(module.weight == 0).item()
                sparsity_list.append(zero_params_num / layer_params_num)
                total_params_num    += layer_params_num
                total_zeros_num     += zero_params_num
                
        return (total_zeros_num / total_params_num, sparsity_list)
