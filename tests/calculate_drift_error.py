import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import os
from network import InfModel
from datetime import datetime
import PlotModule
from Model.PyTorch_CIFAR10.cifar10_models.resnet import resnet18

layer = None

vanilla_model = resnet18(pretrained=True)
output = PlotModule.compute_drift_error(
    vanilla_model, dataset="cifar10", t_seconds=9.33e7, input_layer_names=layer)
print('Output:', output)


# 기본 경로 설정
base_dir = os.getcwd() + '/TestRun/'

# 폴더 이름 (pruning rate 변화)
folder_list = [
    'Test_2024-10-28_15-15_Resnet18_p0.3',
    'Test_2024-10-28_15-22_Resnet18_p0.4',
    'Test_2024-10-28_15-26_Resnet18_p0.5',
    'Test_2024-10-28_15-27_Resnet18_p0.6',
    'Test_2024-10-28_15-32_Resnet18_p0.7',
    'Test_2025-04-01_20-26_Resnet18_p0.8',
    'Test_2025-04-01_20-33_Resnet18_p0.9',
]

# 모델 이름 (fine-tuning 종류)
model_subdirs = ['FT_0.0005_50', 'FT_0.0001_50', 'FT_5e-05_50','FT_1e-05_50', 'FT_1e-06_50']

t_seconds = 9.33e7

# 결과 저장
results = []

for folder in folder_list:
    for model_subdir in model_subdirs:
        folder_path = os.path.join(base_dir, folder, model_subdir)
        model_path = os.path.join(folder_path, 'best_model.pth')
        try:
            print(f"\n[INFO] Evaluating: {folder}/{model_subdir}")
            model = torch.load(model_path, map_location='cpu')

            # Get all 4 drift error ratios
            output1, output2, output3, output4, output5, output6, output7  = PlotModule.compute_drift_error(
                model, dataset="cifar10", t_seconds=t_seconds, input_layer_names=layer
            )

            results.append({
                'Folder': folder,
                'FineTuning': model_subdir,
                'sum_deltaG': output1,
                'sum_deltaG_gdc': output2,
                'sum_deltaG_G0': output3,
                'sum_deltaG_G0_gdc': output4,  
                'sum_deltaG_sum_G0': output5,
                'sum_deltaG_sum_G0_gdc': output6,   
                'alpha': output7,
            })

        except Exception as e:
            print(f"[ERROR] Failed on {folder}/{model_subdir}: {e}")

# 데이터프레임 변환
df = pd.DataFrame(results)

# 엑셀로 저장
timestamp = datetime.now().strftime('%Y%m%d_%H%M')
excel_name = f"DriftErrorData_{timestamp}.xlsx"
df.to_excel(excel_name, index=False)