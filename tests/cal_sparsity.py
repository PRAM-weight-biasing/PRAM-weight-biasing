import torch
import torch.nn as nn
import os


def cal_global_sparsity(model) -> float:
    
        total_params = 0
        zero_params = 0
        # Iterate over all modules and sum total and zeroed-out parameters
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                total_params += module.weight.numel()
                zero_params += torch.sum(module.weight == 0).item()
        global_sparsity = zero_params / total_params  # Sparsity ratio

        print(f'--- Global sparsity of model ---')
        print(f"Global Sparsity: {global_sparsity:.2%}\n")
        return global_sparsity
    
    
    
# 기본 경로 설정
base_dir = os.getcwd() + '/TestRun/'

# 폴더 이름 (pruning rate 변화)
folder_list = [
    'Test_2024-10-28_15-15_Resnet18_p0.3',
    # 'Test_2024-10-28_15-22_Resnet18_p0.4',
    # 'Test_2024-10-28_15-26_Resnet18_p0.5',
    # 'Test_2024-10-28_15-27_Resnet18_p0.6',
    # 'Test_2024-10-28_15-32_Resnet18_p0.7',
    # 'Test_2025-04-01_20-26_Resnet18_p0.8',
    # 'Test_2025-04-01_20-33_Resnet18_p0.9',
]

# 모델 이름 (fine-tuning 종류)
model_subdirs = [
    'FT_0.0005_50', 
    'FT_0.0001_50', 
    'FT_5e-05_50',
    'FT_1e-05_50', 
    'FT_1e-06_50',
    ]


for folder in folder_list:
    for model_subdir in model_subdirs:
        folder_path = os.path.join(base_dir, folder, model_subdir)
        model_path = os.path.join(folder_path, 'best_model.pth')
        model = torch.load(model_path, map_location='cpu')
        print(folder_path)
        
        cal_global_sparsity(model)