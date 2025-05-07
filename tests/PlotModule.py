import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
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
    
    model = model.to("cpu")   # remove if error occurs
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

# plot weight distribution : layer-wise
def plot_weight_dist_all(model):
    # 2. 레이어별 weight 가져오기
    weights = {}
    for name, param in model.named_parameters():
        if ('weight' in name) and ('bn' not in name) and ('downsample.1' not in name):
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
    """Compare weight distributions of same layer from two models """
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
    
    # Determine the max y-axis limit
    max_freq = max(n1.max(), n2.max())
    
    # Set y-axis limits to be the same
    ax1.set_ylim(0, max_freq * 1.1)  # Adding 10% padding
    ax2.set_ylim(0, max_freq * 1.1)
    
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
    
def save_weights_for_matlab(model, layer_name, filename="weights.mat"):
    """Extracts weights from a specific layer and saves them in a .mat file for MATLAB."""
    weights = None
    for name, param in model.named_parameters():
        if layer_name in name:
            weights = param.data.cpu().numpy().flatten()  # 1D 배열로 변환
            break
    
    if weights is not None:
        scipy.io.savemat(filename, {"weights": weights})
        print(f"Saved weights to {filename}")
    else:
        print("Layer not found!")
        

def compute_total_drift_error(model, dataset, gdc=True, ideal_io=True, t_seconds=3600*24*7):
    """
    전체 모델 weight를 analog로 변환하고,
    드리프트를 적용한 후 초기 conductance와의 차이를 계산합니다.
    
    Usage:
        t_seconds = 9.33e7
        compute_total_drift_error(finetuned_model, dataset="cifar10", t_seconds=t_seconds)
        
    """
    
    inf_model = InfModel(model, dataset, noise_list=[0,0])
    rpu_config = inf_model.SetConfig(gdc=gdc, ideal_io=ideal_io)
    analog_model = inf_model.ConvertModel(gdc=gdc, ideal_io=ideal_io)

    noise_model = rpu_config.noise_model
    g_converter = noise_model.g_converter

    all_g_init = []
    all_g_drifted = []
    all_delta_g = []
    total_drift_error = 0.0
    total_drift_error_ratio = 0.0

    for name, param in analog_model.named_parameters():
        if "weight" in name and len(param.size()) > 1:
            weights = param.data.cpu()

            # Conductance 변환
            (gp, gm), _ = g_converter.convert_to_conductances(weights)

            # Programming noise (optional, realistic)
            gp_prog = noise_model.apply_programming_noise_to_conductance(gp)
            gm_prog = noise_model.apply_programming_noise_to_conductance(gm)
            # gp_prog = gp
            # gm_prog = gm

            # Drift coefficient 생성
            nu_gp = noise_model.generate_drift_coefficients(gp_prog)
            nu_gm = noise_model.generate_drift_coefficients(gm_prog)

            # Drift 적용
            gp_drifted = noise_model.apply_drift_noise_to_conductance(gp_prog, nu_gp, t_seconds)
            gm_drifted = noise_model.apply_drift_noise_to_conductance(gm_prog, nu_gm, t_seconds)

            # ΔG 계산
            g_init = gp_prog - gm_prog
            g_drifted = gp_drifted - gm_drifted
            delta_g = torch.abs(g_init - g_drifted)
            delta_g_ratio = torch.abs(delta_g / g_init)
            
            all_g_init.append(g_init.flatten())
            all_g_drifted.append(g_drifted.flatten())
            all_delta_g.append(delta_g.flatten())
            
            # total conductance
            total_g_init += torch.sum(g_init).item()
            total_g_drifted += torch.sum(g_drifted).item()
            
            # total drift error
            drift_error = torch.sum(torch.abs(g_init - g_drifted)).item()
            total_drift_error += drift_error
            
            # total drift error ratio - exclude zero conductance
            nonzero_mask = g_init != 0
            drift_error_ratio = torch.sum(delta_g[nonzero_mask] / g_init[nonzero_mask].abs()).item()
            total_drift_error_ratio += drift_error_ratio
            
    # 전체 G 및 ΔG 모으기
    g_init_all = torch.cat(all_g_init).numpy()
    g_drifted_all = torch.cat(all_g_drifted).numpy()
    delta_g_all = torch.cat(all_delta_g).numpy()
    
    # for global drift compensation
    alpha = total_g_drifted / total_g_init
    total_drift_error_ratio_gdc = total_drift_error_ratio * alpha

    # ==== 출력: 통계 요약 ====
    print("[TOTAL DRIFT STATS]")
    print(f"ΔG mean: {delta_g_all.mean():.6f}")
    print(f"ΔG std : {delta_g_all.std():.6f}")
    print(f"ΔG max : {delta_g_all.max():.6f}")
    print(f"ΔG 99th percentile: {np.percentile(delta_g_all, 99):.6f}")
    print(f"Total drift error after {t_seconds}sec: {total_drift_error:.4f}")
    print(f"Total drift error ratio after {t_seconds}sec: {total_drift_error_ratio:.4f}")
    print(f"alpha: {alpha:.6f}")
    print(f"Total drift error ratio after {t_seconds}sec (GDC): {total_drift_error_ratio_gdc:.4e}")

# plot conductance distribution : before & after overlapped, layer-wise
def compute_and_plot_network_drift(model, dataset, gdc=True, ideal_io=True, t_seconds=3600*24*7):
    inf_model = InfModel(model, dataset, noise_list=[0, 0])
    rpu_config = inf_model.SetConfig(gdc=gdc, ideal_io=ideal_io)
    analog_model = inf_model.ConvertModel(gdc=gdc, ideal_io=ideal_io)

    noise_model = rpu_config.noise_model
    g_converter = noise_model.g_converter

    all_g_init = []
    all_g_drifted = []
    all_delta_g = []

    for name, param in analog_model.named_parameters():
        if "weight" in name and len(param.size()) > 1:
            weights = param.data.cpu()

            (gp, gm), _ = g_converter.convert_to_conductances(weights)

            gp_prog = noise_model.apply_programming_noise_to_conductance(gp)
            gm_prog = noise_model.apply_programming_noise_to_conductance(gm)

            nu_gp = noise_model.generate_drift_coefficients(gp_prog)
            nu_gm = noise_model.generate_drift_coefficients(gm_prog)

            gp_drifted = noise_model.apply_drift_noise_to_conductance(gp_prog, nu_gp, t_seconds)
            gm_drifted = noise_model.apply_drift_noise_to_conductance(gm_prog, nu_gm, t_seconds)

            g_init = gp_prog - gm_prog
            g_drifted = gp_drifted - gm_drifted
            delta_g = torch.abs(g_init - g_drifted)

            all_g_init.append(g_init.flatten())
            all_g_drifted.append(g_drifted.flatten())
            all_delta_g.append(delta_g.flatten())

    # 전체 G 및 ΔG 모으기
    g_init_all = torch.cat(all_g_init).numpy()
    g_drifted_all = torch.cat(all_g_drifted).numpy()
    delta_g_all = torch.cat(all_delta_g).numpy()

    # ==== 출력: 통계 요약 ====
    print("[TOTAL DRIFT STATS]")
    print(f"ΔG mean: {delta_g_all.mean():.6f}")
    print(f"ΔG std : {delta_g_all.std():.6f}")
    print(f"ΔG max : {delta_g_all.max():.6f}")
    print(f"ΔG 99th percentile: {np.percentile(delta_g_all, 99):.6f}")

    # ==== 시각화 ====
    plt.figure(figsize=(14, 5))

    # # G 분포 비교 : whole network
    # plt.subplot(1, 2, 1)
    # plt.hist(g_init_all, bins=500, alpha=0.7, label='Before Drift')
    # plt.hist(g_drifted_all, bins=500, alpha=0.7, label='After Drift')
    # plt.title("Conductance Distribution (Whole Network)")
    # plt.xlabel("Effective Conductance (Gp - Gm)")
    # plt.ylabel("Frequency")
    # plt.legend()
    # plt.grid(True)
    # plt.ylim(0, 500000)
    
    # ==== G 분포 비교: 레이어별 ====
    num_layers = len(all_g_init)
    cols = 3
    rows = (num_layers + 1) // cols

    plt.figure(figsize=(cols * 6, rows * 4))
    for i in range(num_layers):
        g_before = all_g_init[i].numpy()
        g_after = all_g_drifted[i].numpy()

        plt.subplot(rows, cols, i + 1)
        plt.hist(g_before, bins=150, alpha=0.7, label='Before Drift')
        plt.hist(g_after, bins=150, alpha=0.7, label='After Drift')
        plt.title(f"G Distribution (Layer {i+1})")
        plt.xlabel("Effective Conductance (Gp - Gm)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.legend()
        plt.yscale('log')

    plt.tight_layout()
    plt.show()

    # ΔG 분포
    plt.subplot(1, 2, 2)
    plt.hist(delta_g_all, bins=500, alpha=0.8, color='darkred')
    plt.title("ΔG (Drift Error) Distribution - Whole Network")
    plt.xlabel("|ΔG|")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.ylim(0, 500000)
    plt.tight_layout()
    plt.show()
    
    
def __compute_layer_drift_error(model, dataset, layer, gdc=True, ideal_io=True, t_seconds=3600*24*7):

    inf_model = InfModel(model, dataset, noise_list=[0,0])
    rpu_config = inf_model.SetConfig(gdc=gdc, ideal_io=ideal_io)
    analog_model = inf_model.ConvertModel(gdc=gdc, ideal_io=ideal_io)

    noise_model = rpu_config.noise_model
    g_converter = noise_model.g_converter
    
    all_g_init = []
    all_g_drifted = []
    all_delta_g = []
    total_drift_error = 0.0
    total_drift_error_ratio = 0.0
    total_g_init = 0.0
    total_g_drifted = 0.0
    total_nu = 0.0

    for name, param in analog_model.named_parameters():
        if 'weight' in name and name.startswith(layer) and len(param.size()) > 1:
            print(name)
            weights = param.data.cpu()

            # Conductance 변환
            (gp, gm), _ = g_converter.convert_to_conductances(weights)

            # Programming noise (optional, realistic)
            gp_prog = noise_model.apply_programming_noise_to_conductance(gp)
            gm_prog = noise_model.apply_programming_noise_to_conductance(gm)
            # gp_prog = gp
            # gm_prog = gm

            # Drift coefficient 생성
            nu_gp = noise_model.generate_drift_coefficients(gp_prog)
            nu_gm = noise_model.generate_drift_coefficients(gm_prog)

            # Drift 적용
            gp_drifted = noise_model.apply_drift_noise_to_conductance(gp_prog, nu_gp, t_seconds)
            gm_drifted = noise_model.apply_drift_noise_to_conductance(gm_prog, nu_gm, t_seconds)

            # ΔG 계산
            g_init = gp_prog - gm_prog
            g_drifted = gp_drifted - gm_drifted
            delta_g = torch.abs(g_init - g_drifted)
            # delta_g_ratio = torch.abs(delta_g / g_init)
            
            all_g_init.append(g_init.flatten())
            all_g_drifted.append(g_drifted.flatten())
            all_delta_g.append(delta_g.flatten())
            
            # total drift error
            drift_error = torch.sum(torch.abs(g_init - g_drifted)).item()
            total_drift_error += drift_error
            
            # total conductance
            # total_g_init += torch.sum(g_init).item()
            # total_g_drifted += torch.sum(g_drifted).item()
            total_g_init += torch.sum(torch.abs(g_init)).item()
            total_g_drifted += torch.sum(torch.abs(g_drifted)).item()

            # exclude zero conductance
            nonzero_mask = g_init != 0
            drift_error_ratio = torch.sum(delta_g[nonzero_mask] / g_init[nonzero_mask].abs()).item()
            total_drift_error_ratio += drift_error_ratio
            
            # calculate sum of nu
            total_nu += torch.sum(nu_gp).item() + torch.sum(nu_gm).item()
            
    # 전체 G 및 ΔG 모으기
    g_init_all = torch.cat(all_g_init).numpy()
    g_drifted_all = torch.cat(all_g_drifted).numpy()
    delta_g_all = torch.cat(all_delta_g).numpy()
    
    # for global drift compensation
    alpha = total_g_drifted / total_g_init
    total_drift_error_ratio_gdc = total_drift_error_ratio * alpha
    total_drift_error_gdc = total_drift_error * alpha
        
    # ==== 출력: 통계 요약 ====
    print("[TOTAL DRIFT STATS]")
    print(f"ΔG mean: {delta_g_all.mean():.6f}")
    print(f"ΔG std : {delta_g_all.std():.6f}")
    print(f"ΔG max : {delta_g_all.max():.6f}")
    # print(f"ΔG 99th percentile: {np.percentile(delta_g_all, 99):.6f}")
    print(f"Total drift error after {t_seconds}sec: {total_drift_error:.4e}")
    print(f"Total drift error ratio after {t_seconds}sec: {total_drift_error_ratio:.4e}")
    print(f"alpha: {alpha:.6f}")
    print(f"Total drift error ratio after {t_seconds}sec (GDC): {total_drift_error_ratio_gdc:.4e}")
    
    return total_drift_error, total_drift_error_gdc,total_drift_error_ratio, total_drift_error_ratio_gdc

def __compute_total_drift_error_r1(model, dataset, gdc=True, ideal_io=True, t_seconds=3600*24*7):
    """ 
    Usage:
        t_seconds = 9.33e7
        compute_total_drift_error(finetuned_model, dataset="cifar10", t_seconds=t_seconds)
        
    """
       
    import myModule
    myModule.fix_seed(42)
    
    inf_model = InfModel(model, dataset, noise_list=[0,0])
    rpu_config = inf_model.SetConfig(gdc=gdc, ideal_io=ideal_io)
    analog_model = inf_model.ConvertModel(gdc=gdc, ideal_io=ideal_io)

    noise_model = rpu_config.noise_model
    g_converter = noise_model.g_converter

    total_g_init = 0.0
    total_g_drifted = 0.0
    total_drift_err = 0.0
    total_drift_err_ratio = 0.0
    total_nu = 0.0

    for name, param in analog_model.named_parameters():
        if "weight" in name and len(param.size()) > 1:
            weights = param.data.cpu()

            # weight to conductance
            (gp, gm), _ = g_converter.convert_to_conductances(weights)

            # Programming noise (optional, realistic)
            # gp_prog = noise_model.apply_programming_noise_to_conductance(gp)
            # gm_prog = noise_model.apply_programming_noise_to_conductance(gm)
            gp_prog = gp
            gm_prog = gm

            # Drift coefficient
            nu_gp = noise_model.generate_drift_coefficients(gp_prog)
            nu_gm = noise_model.generate_drift_coefficients(gm_prog)

            # after drift 
            gp_drifted = noise_model.apply_drift_noise_to_conductance(gp_prog, nu_gp, t_seconds)
            gm_drifted = noise_model.apply_drift_noise_to_conductance(gm_prog, nu_gm, t_seconds)
            
            # calculate delta G
            delta_gp = gp_prog - gp_drifted 
            delta_gm = gm_prog - gm_drifted
            
            # Create masks for non-zero conductances
            nonzero_gp = (gp_prog != 0)
            nonzero_gm = (gm_prog != 0)
            delta_gp_ratio = (delta_gp[nonzero_gp] / gp_prog[nonzero_gp])
            delta_gm_ratio = (delta_gm[nonzero_gm] / gm_prog[nonzero_gm])
            
            # total conductance = Gp + Gm
            total_g_init += torch.sum(gp_prog).item() + torch.sum(gm_prog).item()
            total_g_drifted += torch.sum(gp_drifted).item() + torch.sum(gm_drifted).item()
            
            # total drift error = (Gp-Gp_drifted) + (Gm-Gm_drifted)
            drift_err = torch.sum(delta_gp).item() + torch.sum(delta_gm).item()
            total_drift_err += drift_err
            
            # total drift error ratio = drift_err / G
            drift_err_ratio = torch.sum(delta_gp_ratio).item() + torch.sum(delta_gm_ratio).item()
            total_drift_err_ratio += drift_err_ratio
            
            # calculate sum of nu
            total_nu += torch.sum(nu_gp).item() + torch.sum(nu_gm).item()
        
    # for global drift compensation
    alpha = total_g_drifted / total_g_init
    total_drift_err_gdc = total_g_init - (total_g_drifted / alpha)
    total_err_over_init = total_drift_err / total_g_init
    total_err_over_init_gdc = total_drift_err_gdc / total_g_init

    output = [
        total_drift_err,           # sum[G0 - G]                = sum[ΔG]
        total_drift_err_gdc,       # sum[G0 - G/alpha]
        total_drift_err_ratio,     # sum[(G0 - G) / G0]         = sum[ΔG / G0]
        total_err_over_init,       # sum[G0 - G] / sum[G0]      = sum[ΔG] / sum[G0]
        total_err_over_init_gdc,   # sum[G0 - G/alpha] / sum[G0] 
        alpha,                     # sum[G] / sum[G0] 
         
    ]
    
    # ==== 출력: 통계 요약 ====
    print("[TOTAL DRIFT STATS]")
    print(f"Total drift error after {t_seconds}sec: {total_drift_err:.4e}")
    print(f"Total drift error ratio after {t_seconds}sec: {total_drift_err_ratio:.4e}")
    print(f"alpha: {alpha:.5f}")
           
    return output


def compute_drift_error(model, dataset, gdc=True, ideal_io=True, t_seconds=36*30*24*3600, input_layer_names=None):
    """ 
    Compute total drift error, both before and after global drift compensation (GDC).
    """

    import myModule
    # myModule.fix_seed(42)

    inf_model = InfModel(model, dataset, noise_list=[0, 0])
    rpu_config = inf_model.SetConfig(gdc=gdc, ideal_io=ideal_io)
    analog_model = inf_model.ConvertModel(gdc=gdc, ideal_io=ideal_io)

    noise_model = rpu_config.noise_model
    g_converter = noise_model.g_converter

    total_g_init = 0.0
    total_g_drifted = 0.0
    total_drift_err = 0.0
    total_drift_err_ratio = 0.0
    total_nu = 0.0

    # 1. Compute initial, drifted, and uncorrected errors
    drift_data = []  # Save per-layer conductances for later GDC

    for name, param in analog_model.named_parameters():
        
        # 조건: weight 파라미터, 2차원 이상, 그리고 레이어 이름 매칭
        if 'weight' in name and len(param.size()) > 1:
            if input_layer_names is not None:
                matched = any(name.startswith(layer) for layer in input_layer_names)
                if not matched:
                    continue  # 해당 레이어가 아니면 skip

        # if "weight" in name and len(param.size()) > 1:
            weights = param.data.cpu()
            
            # Convert weights to conductances
            (gp, gm), _ = g_converter.convert_to_conductances(weights)

            # Programming noise (optional, realistic)
            # gp_prog = noise_model.apply_programming_noise_to_conductance(gp)
            # gm_prog = noise_model.apply_programming_noise_to_conductance(gm)
            gp_prog = gp
            gm_prog = gm

            # Drift coefficient generation
            nu_gp = noise_model.generate_drift_coefficients(gp_prog)
            nu_gm = noise_model.generate_drift_coefficients(gm_prog)

            gp_drifted = noise_model.apply_drift_noise_to_conductance(gp_prog, nu_gp, t_seconds)
            gm_drifted = noise_model.apply_drift_noise_to_conductance(gm_prog, nu_gm, t_seconds)
            
            # Calculate delta G
            delta_gp = gp_prog - gp_drifted 
            delta_gm = gm_prog - gm_drifted

            # Calculate delta G ratio (for non-zero conductances)
            nonzero_gp = (gp_prog != 0)
            nonzero_gm = (gm_prog != 0)

            delta_gp_ratio = (delta_gp[nonzero_gp] / gp_prog[nonzero_gp])
            delta_gm_ratio = (delta_gm[nonzero_gm] / gm_prog[nonzero_gm])

            # Total values
            total_g_init += torch.sum(gp_prog).item() + torch.sum(gm_prog).item()
            total_g_drifted += torch.sum(gp_drifted).item() + torch.sum(gm_drifted).item()
            total_drift_err += torch.sum(delta_gp).item() + torch.sum(delta_gm).item()
            total_drift_err_ratio += torch.sum(delta_gp_ratio).item() + torch.sum(delta_gm_ratio).item()
            total_nu += torch.sum(nu_gp).item() + torch.sum(nu_gm).item()

            # Save for later GDC re-evaluation
            drift_data.append((gp_prog, gm_prog, gp_drifted, gm_drifted))

    # 2. Global drift compensation (GDC)
    alpha = total_g_drifted / total_g_init
    total_drift_err_gdc = 0.0
    total_drift_err_ratio_gdc = 0.0

    for gp_prog, gm_prog, gp_drifted, gm_drifted in drift_data:
        # Calculate GDC-corrected delta G
        gp_drifted_gdc = gp_drifted / alpha
        gm_drifted_gdc = gm_drifted / alpha

        delta_gp_gdc = gp_prog - gp_drifted_gdc
        delta_gm_gdc = gm_prog - gm_drifted_gdc

        total_drift_err_gdc += torch.sum(torch.abs(delta_gp_gdc)).item() + torch.sum(torch.abs(delta_gm_gdc)).item()
        # total_drift_err_gdc += torch.sum(delta_gp_gdc).item() + torch.sum(delta_gm_gdc).item()  # 상쇄되는 경우

        # Calculate GDC-corrected delta G ratio (for non-zero conductances)
        nonzero_gp = (gp_prog != 0)
        nonzero_gm = (gm_prog != 0)

        delta_gp_gdc_ratio = (delta_gp_gdc[nonzero_gp] / gp_prog[nonzero_gp])
        delta_gm_gdc_ratio = (delta_gm_gdc[nonzero_gm] / gm_prog[nonzero_gm])
        total_drift_err_ratio_gdc += torch.sum(torch.abs(delta_gp_gdc_ratio)).item() + torch.sum(torch.abs(delta_gm_gdc_ratio)).item()
        # total_drift_err_ratio_gdc += torch.sum(delta_gp_gdc_ratio).item() + torch.sum(delta_gm_gdc_ratio).item() # 상쇄되는 경우

    # 3. Summary stats
    total_err_over_init = total_drift_err / total_g_init
    total_err_over_init_gdc = total_drift_err_gdc / total_g_init

    output = [
        total_drift_err,             # 1
        total_drift_err_gdc,         # 2
        total_drift_err_ratio,       # 3
        total_drift_err_ratio_gdc,   # 4
        total_err_over_init,         # 5
        total_err_over_init_gdc,     # 6
        alpha,                       # 7   
    ]

    # === 출력 ===
    print("[TOTAL DRIFT STATS]")
    print(f"Target layers: {'ALL' if input_layer_names is None else input_layer_names}")
    print(f"Total drift error after {t_seconds:.1f} sec: {total_drift_err:.4e}")
    print(f"Total drift error ratio (sum of ΔG/G₀): {total_drift_err_ratio:.4e}")
    print(f"alpha (G_drifted / G_init): {alpha:.5f}")
    # print(f"GDC-corrected drift error: {total_drift_err_gdc:.4e}")
    # print(f"GDC-corrected drift error ratio (ΔG/G₀): {total_drift_err_ratio_gdc:.4e}")

    return output
    