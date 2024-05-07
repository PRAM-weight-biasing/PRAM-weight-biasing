import torch
import random
import os

# import customized files
from network import PruneModel

### =============================================================


# Setting
random.seed(777)
torch.manual_seed(777)

dir_name = os.getcwd() + '/TestRun/'
# ===========================================
""" need to change """
test_time = "Test_2024-05-07 16:27:42" 
# ===========================================
folder_path = dir_name + test_time
model_name = 'best_model.pth'

prune_percentage = 0.3

# local pruning : prune each layer
PruneTest1 = PruneModel(prune_percentage, model_name, folder_path)
PruneTest1.local_pruning()
sparsity_list = PruneTest1.cal_local_sparsity('local_pruned_model.pth')
PruneTest1.Vis_local_w()

# global pruning : prune the whole network
PruneTest2 = PruneModel(prune_percentage, model_name, folder_path)
PruneTest2.global_pruning()
sparsity_list = PruneTest1.cal_global_sparsity('global_pruned_model.pth')
PruneTest1.Vis_global_w()
