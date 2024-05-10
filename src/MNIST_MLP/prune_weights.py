import torch
import torch.nn as nn
import numpy as np

class PruneModel():
    def __init__(self, prune_pct: float, model: nn.Sequential, save_dir: str):
        self.save_dir       = save_dir
        self.model_vanilla  = model
        self.prune_percent  = prune_pct
        self.model_pruned   = None

    def local_pruning(self) -> None:
        pass

    def global_pruning(self) -> None:
        pass

    def calc_local_sparsity(self) -> list:
        pass

    def calc_global_sparsity(self) -> float:
        pass

    def vis_local_weight_distribution(self) -> None:
        pass

    def vis_global_weight_distribution(self) -> None:
        pass