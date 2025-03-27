import torch
import matplotlib.pyplot as plt
import numpy as np
from network import InfModel


def analyze_model_weights(model, folder_name):
    weights = []
    
    for param in model.parameters():
        if param.requires_grad:  # 학습 가능한 가중치만 선택
            weights.append(param.data.view(-1))  # 1D 텐서로 변환
    
    weights = torch.cat(weights)  # 모든 가중치를 하나의 벡터로 결합
    
    abs_min = torch.min(torch.abs(weights)).item()
    abs_max = torch.max(torch.abs(weights)).item()
    
    # 0보다 큰 값 중 최소값 찾기
    positive_weights = weights[weights > 0]  # 0보다 큰 값 필터링
    min_positive = positive_weights.min().item() if positive_weights.numel() > 0 else None  # 값이 있을 때만 최소값 계산
    
    # 0보다 작은 값 중 최소값 찾기
    negative_weights = weights[weights < 0]  
    min_negative = negative_weights.max().item() if negative_weights.numel() > 0 else None  # 값이 있을 때만 최소값 계산


    print(f"Absolute Min: {abs_min}, Absolute Max: {abs_max:.5f}")
    print(f"Min Positive Value: {min_positive}, Min Negative Value: {min_negative}")
    
    # 가중치 분포 시각화
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1행 2열 서브플롯
    
    # 전체 제목 (중앙)
    fig.suptitle(folder_name, fontsize=16, fontweight='bold')  

    # Linear scale subplot
    axes[0].hist(weights.cpu().numpy(), bins=200, alpha=0.75, color='blue')
    axes[0].set_xlabel("Weight Values")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Weight Distribution (Linear Scale)")
    axes[0].grid(True)

    # Log scale subplot
    axes[1].hist(weights.cpu().numpy(), bins=200, alpha=0.75, color='blue')
    axes[1].set_xlabel("Weight Values")
    axes[1].set_ylabel("Frequency (log scale)")
    axes[1].set_title("Weight Distribution (Log Scale)")
    axes[1].set_yscale('log')  # 로그 스케일 적용
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
    
    return abs_min, abs_max

# cnn / fc legend
def extract_weights(model):
    conv_weights = []
    fc_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.size()) > 1:  # Consider only convolutional and fully connected layers
            if len(param.size()) == 4:  # Convolutional layers
                conv_weights.append(param.data.cpu().numpy().flatten())
            elif len(param.size()) == 2:  # Fully connected layers
                fc_weights.append(param.data.cpu().numpy().flatten())
    return conv_weights, fc_weights


def plot_weights(model, model_name):
    conv_weights, fc_weights = extract_weights(model)
    
     # Flatten the weights list
    all_conv_weights = np.concatenate(conv_weights) if conv_weights else np.array([])
    all_fc_weights = np.concatenate(fc_weights) if fc_weights else np.array([])

    # 0보다 큰 값 중 최소값 찾기
    positive_weights = all_conv_weights[all_conv_weights > 0]  # NumPy 배열에서 필터링
    min_posi = positive_weights.min() if positive_weights.size > 0 else None  # 값이 있을 때만 최소값 계산
    
    # 0보다 작은 값 중 최대값 찾기
    negative_weights = all_conv_weights[all_conv_weights < 0]
    min_nega = negative_weights.max() if negative_weights.size > 0 else None  # 값이 있을 때만 최대값 계산


    print(f"Min Positive Value: {min_posi}, Min Negative Value: {min_nega}")
    
    # Plot the distribution of weights
    plt.hist(all_conv_weights, bins=100, alpha=0.7, label="conv")
    plt.hist(all_fc_weights, bins=100, alpha=0.7, label="fc")
    plt.title(f'{model_name}')
    plt.xlabel('Weight value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, axis='y')
    plt.yscale('log')
    plt.show()
    
    
# Conductance 변환 함수 (FC / Conv 분리)
def convert_weights_to_conductance(model, rpu_config):
    conductance_list_fc = []
    conductance_list_conv = []

    min_gp_fc, min_gm_fc = None, None
    min_gp_conv, min_gm_conv = None, None

    for name, param in model.named_parameters():
        if "weight" in name and len(param.size()) > 1:  # Conv / FC layer만 처리
            weights = param.data.cpu()
            
            # Conductance 변환 수행
            conductance_pair = rpu_config.noise_model.g_converter.convert_to_conductances(weights)
            gp, gm = conductance_pair[0]  # conductance_pair는 (gp, gm), params 반환

            # 0이 아닌 값들만 필터링
            gp_nonzero = gp[gp > 0]  # 0보다 큰 값만 선택
            gm_nonzero = gm[gm > 0]  # 0보다 큰 값만 선택

            if "fc" in name or weights.shape[0] <= 10:  # Fully Connected Layer
                conductance_list_fc.append((gp.flatten(), gm.flatten()))

                # 최소값 업데이트
                if gp_nonzero.numel() > 0:
                    min_gp_fc = gp_nonzero.min().item() if min_gp_fc is None else min(min_gp_fc, gp_nonzero.min().item())
                if gm_nonzero.numel() > 0:
                    min_gm_fc = gm_nonzero.min().item() if min_gm_fc is None else min(min_gm_fc, gm_nonzero.min().item())

            else:  # Convolutional Layer
                conductance_list_conv.append((gp.flatten(), gm.flatten()))

                # 최소값 업데이트
                if gp_nonzero.numel() > 0:
                    min_gp_conv = gp_nonzero.min().item() if min_gp_conv is None else min(min_gp_conv, gp_nonzero.min().item())
                if gm_nonzero.numel() > 0:
                    min_gm_conv = gm_nonzero.min().item() if min_gm_conv is None else min(min_gm_conv, gm_nonzero.min().item())

    print(f"Min Nonzero Gp (FC): {min_gp_fc}, Min Nonzero Gm (FC): {min_gm_fc}")
    print(f"Min Nonzero Gp (Conv): {min_gp_conv}, Min Nonzero Gm (Conv): {min_gm_conv}")

    return conductance_list_fc, conductance_list_conv

# Conductance distribution plot
def plot_conductance_distribution(model, model_name, gdc:bool, ideal_io:bool):
    inf_model = InfModel(model, "cifar10")
    rpu_config = inf_model.SetConfig(gdc=gdc, ideal_io=ideal_io)
    analog_model = inf_model.ConvertModel(gdc=gdc, ideal_io=ideal_io)

    conductance_list_fc, conductance_list_conv = convert_weights_to_conductance(analog_model, rpu_config)

    gp_fc_all = np.concatenate([gp for gp, _ in conductance_list_fc]) if conductance_list_fc else np.array([])
    gm_fc_all = np.concatenate([gm for _, gm in conductance_list_fc]) if conductance_list_fc else np.array([])

    gp_conv_all = np.concatenate([gp for gp, _ in conductance_list_conv]) if conductance_list_conv else np.array([])
    gm_conv_all = np.concatenate([gm for _, gm in conductance_list_conv]) if conductance_list_conv else np.array([])

    fig, axes = plt.subplots(1,2, figsize=(12, 5))
    # 전체 제목 (중앙)
    fig.suptitle(model_name, fontsize=14, fontweight='bold')  

    # FC 레이어 Conductance Plot
    axes[0].hist(gp_fc_all, bins=150, alpha=1, label="Gp (FC)")
    axes[0].hist(-gm_fc_all, bins=150, alpha=0.7, label="Gm (FC)")
    axes[0].set_title('G Distribution (FC)')
    axes[0].set_xlabel('Conductance Value')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, axis='y')
    axes[0].set_yscale('log')

    # Conv 레이어 Conductance Plot
    axes[1].hist(gp_conv_all, bins=500, alpha=1, label="Gp (Conv)")
    axes[1].hist(-gm_conv_all, bins=500, alpha=0.7, label="Gm (Conv)")
    axes[1].set_title('G Distribution (Conv)')
    axes[1].set_xlabel('Conductance Value')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, axis='y')
    axes[1].set_yscale('log')
    # axes[1].set_xlim(-0.1,10)

    plt.tight_layout()
    plt.show()
    
def plot_weight_dist_all(model):
    # 2. 레이어별 weight 가져오기
    weights = {}
    for name, param in model.named_parameters():
        if ('weight' in name):
            weights[name] = param.detach().cpu().numpy().flatten()

    # 3. Subplot 설정
    num_layers = len(weights)
    cols = 5  # 한 줄에 표시할 subplot 개수
    rows = (num_layers + cols - 1) // cols  # 필요한 행 수 계산

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes = axes.flatten()  # 1D 배열로 변환하여 인덱싱 가능하게 만들기

    # 4. 각 레이어별 weight 분포 그리기
    for idx, (name, w) in enumerate(weights.items()):
        axes[idx].hist(w, bins=100, alpha=0.7)
        axes[idx].set_title(f"{name}")  # 레이어 이름 추가
        axes[idx].set_xlabel("Weight Values")
        axes[idx].set_ylabel("Frequency")
        axes[idx].grid(True)

    # 나머지 빈 subplot 숨기기
    for idx in range(num_layers, len(axes)):
        axes[idx].axis('off')

    # 5. 그래프 표시
    plt.tight_layout()
    plt.show()

def plot_weight_module(model, module_name):
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.size()) > 1:  # Consider only convolutional and fully connected layers
            if len(param.size()) == 4:  # Convolutional layers
                if module_name in name :
                    weights = param.data.cpu().numpy().flatten()
                    plt.hist(weights, bins=150, alpha=0.7, label=f'{name}')
            

    plt.title('Layer 1.0 Convolutional weights (After)')
    plt.xlabel('Weight value')
    plt.ylabel('Frequency')
    plt.legend(loc='upper left', bbox_to_anchor=(1.0,1.0))
    plt.grid(True, axis='y')
    # plt.yscale('log')
    plt.show()
    
def plot_weight_comparison(model1, model2, layer_name, title1="Vanilla", title2="Finetuned"):
    """Compare weight distributions of same layer from two models with synchronized y-axes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt.rcParams.update({'font.size': 12})
    
    # Get weights for both models
    weights1, weights2 = None, None
    
    # Get weights from first model
    for name, param in model1.named_parameters():
        if layer_name in name:
            weights1 = param.data.cpu().numpy().flatten()
            
    # Get weights from second model
    for name, param in model2.named_parameters():
        if layer_name in name:
            weights2 = param.data.cpu().numpy().flatten()
    
    # Plot histograms and get their max frequencies
    n1, _, _ = ax1.hist(weights1, bins=150, alpha=0.7)
    n2, _, _ = ax2.hist(weights2, bins=150, alpha=0.7)
    
    # Set common y limit
    ymax = max(n1.max(), n2.max())
    ax1.set_ylim(0, ymax * 1.1)  # Add 10% padding
    ax2.set_ylim(0, ymax * 1.1)
    
    # Sync y-axis ticks
    yticks = ax1.get_yticks()
    ax2.set_yticks(yticks)
    
    # Set titles and labels
    ax1.set_title(f'{title1} - {layer_name}')
    ax1.set_xlabel('Weight value')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, axis='y')
    
    ax2.set_title(f'{title2} - {layer_name}')
    ax2.set_xlabel('Weight value')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, axis='y')
    
    plt.suptitle('Weight Distribution Comparison', fontsize=16)
    # plt.tight_layout()
    plt.show()