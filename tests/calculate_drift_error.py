import os
import torch
import numpy as np
import scipy.io

from network import InfModel
import PlotModule

from Model.PyTorch_CIFAR10.cifar10_models.resnet import resnet18

# layer-wise drift error

vanilla_model = resnet18(pretrained=True)


# model import
base_dir = os.getcwd() + '/TestRun/'

folder_list = [
    'Test_2024-10-28_15-15_Resnet18_p0.3',
    'Test_2024-10-28_15-22_Resnet18_p0.4',
    'Test_2024-10-28_15-26_Resnet18_p0.5',
    'Test_2024-10-28_15-27_Resnet18_p0.6',
    'Test_2024-10-28_15-32_Resnet18_p0.7',
    'Test_2025-04-01_20-26_Resnet18_p0.8',
    'Test_2025-04-01_20-33_Resnet18_p0.9',
    ]

# fine-tuning rate
model_subdirs = ['FT_0.0005_50', 'FT_0.0001_50', 'FT_5e-05_50','FT_1e-05_50', 'FT_1e-06_50']

# measurement settings
layer = 'fc'
t_seconds = 9.33e7  # 3yr (3 * 365 * 24 * 60 * 60)

# Results
results = []

for folder in folder_list:
    for model_subdir in model_subdirs:
        folder_path = os.path.join(base_dir, folder, model_subdir)
        model_path = os.path.join(folder_path, 'best_model.pth')
        try:
            print(f"\n[INFO] Evaluating: {folder}/{model_subdir}")
            model = torch.load(model_path, map_location='cpu')

            # Get all 4 drift error ratios
            total_drift_error, total_drift_error_gdc, total_drift_error_ratio, total_drift_error_ratio_gdc = compute_layer_drift_error_r1(
                model, dataset="cifar10", layer=layer, t_seconds=t_seconds
            )

            results.append({
                'Folder': folder,
                'FineTuning': model_subdir,
                'total_drift_error': total_drift_error,
                'total_drift_error_gdc': total_drift_error_gdc,
                'total_drift_error_ratio': total_drift_error_ratio,
                'total_drift_error_ratio_gdc': total_drift_error_ratio_gdc,    
            })

        except Exception as e:
            print(f"[ERROR] Failed on {folder}/{model_subdir}: {e}")

# 데이터프레임 변환
df = pd.DataFrame(results)

# 엑셀로 저장
timestamp = datetime.now().strftime('%Y%m%d_%H%M')
excel_name = f"DriftErrorData_{timestamp}.xlsx"
df.to_excel(excel_name, index=False)