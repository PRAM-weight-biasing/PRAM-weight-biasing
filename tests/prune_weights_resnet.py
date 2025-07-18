import torch
import random
import os

# import customized files
import myModule
from network import PruneModel, TrainModel, Vis_Model

from Model.PyTorch_CIFAR10.cifar10_models.resnet import resnet18

### =============================================================


# Setting

# dir_name = os.getcwd() + '/TestRun/'
# ===========================================
# """ need to change """
# test_time = "Test_2024-08-01 11:52:18" 
# # ===========================================
# folder_path = dir_name + test_time
# model_name = 'best_model.pth'

### =============================================================
model_name = resnet18(pretrained=True)

prune_percentage = 0.2
folder_path = myModule.MakeFolder(f'_prune{prune_percentage}')
### =============================================================


# local pruning : prune each layer
PruneTest1 = PruneModel(model_name, folder_path)
PruneTest1.apply_local_pruning(prune_percentage)
sparsity_list1 = PruneTest1.cal_local_sparsity('local_pruned_model.pth')
sparsity_all1 = PruneTest1.cal_global_sparsity('local_pruned_model.pth')
# PruneTest1.Vis_local_w()
# PruneTest1.Vis_global_w()

# global pruning : prune the whole network
# PruneTest2 = PruneModel(model_name, folder_path)
# PruneTest2.apply_global_pruning(prune_percentage)
# sparsity_list2 = PruneTest2.cal_local_sparsity('global_pruned_model.pth')
# sparsity_all2 = PruneTest2.cal_global_sparsity('global_pruned_model.pth')
# PruneTest2.Vis_local_w()
# PruneTest2.Vis_global_w()
