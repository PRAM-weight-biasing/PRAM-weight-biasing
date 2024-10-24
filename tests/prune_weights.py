import torch
import random
import os

# import customized files
import myModule
from network import PruneModel, TrainModel, Vis_Model
from Model.PyTorch_CIFAR10.cifar10_models.resnet import resnet18

# =============================================================

# Import model
model_type = input("Input model type? (MLP/Resnet18) : ")
prune_percentage = float(input('Pruning percentage? (0~1) : '))

if model_type == 'MLP':
    dir_name = os.getcwd() + '/Model/MLP'
    model_name = torch.load(f'{dir_name}/best_model.pth')
    
elif model_type == 'Resnet18':
    model_name = resnet18(pretrained=True)
    
folder_path = myModule.MakeFolder(f'_{model_type}_p{prune_percentage}')


# local pruning : prune each layer
PruneTest1 = PruneModel(model_name, folder_path)
PruneTest1.apply_local_pruning(prune_percentage)
sparsity_list1 = PruneTest1.cal_local_sparsity('local_pruned_model.pth')
sparsity_all1 = PruneTest1.cal_global_sparsity('local_pruned_model.pth')
PruneTest1.Vis_local_w()
PruneTest1.Vis_global_w()

# global pruning : prune the whole network
# PruneTest2 = PruneModel(model_name, folder_path)
# PruneTest2.apply_global_pruning(prune_percentage)
# sparsity_list2 = PruneTest2.cal_local_sparsity('global_pruned_model.pth')
# sparsity_all2 = PruneTest2.cal_global_sparsity('global_pruned_model.pth')
# PruneTest2.Vis_local_w()
# PruneTest2.Vis_global_w()
