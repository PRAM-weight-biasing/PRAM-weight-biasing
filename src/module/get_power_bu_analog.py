# -*- coding: utf-8 -*-
""" Array-only proxy energy with analog model (layer-wise streaming, no percentile). """

import torch
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Linear, BatchNorm2d, Identity
import pandas as pd
import numpy as np
from typing import Dict, Any

from aihwkit.nn import AnalogConv2d, AnalogLinear

# ------------------------------
# helper: flatten conv weight to [out, in*kH*kW], linear is already [out, in]
# ------------------------------
def _weight_matrix_from_module(m: Module) -> torch.Tensor:
    if isinstance(m, Conv2d):
        return m.weight.reshape(m.weight.shape[0], -1)
    elif isinstance(m, Linear):
        return m.weight
    else:
        raise TypeError("Only Conv2d and Linear are supported.")


# ------------------------------
# helper: column-sum of total conductance (sum over rows) -> shape [N_in_cols]
# ------------------------------
def _column_sum_G_total(g_list) -> torch.Tensor:
    """g_list: list of conductance tensors, e.g. [gp, gm] or [g1p, g1m, g2p, g2m, ...]"""
    if len(g_list) == 2:
        gp, gm = g_list
        return gp.sum(dim=0) + gm.sum(dim=0)
    else:
        G_total = torch.zeros(g_list[0].shape[1], device=g_list[0].device, dtype=g_list[0].dtype)
        for g_plus, g_minus in zip(g_list[::2], g_list[1::2]):
            G_total += g_plus.sum(dim=0) + g_minus.sum(dim=0)
        return G_total

# ------------------------------
# helper: remove BatchNorm2d
# ------------------------------
def remove_batchnorm(model: Module) -> Module:
    """Replace all nn.BatchNorm2d layers with Identity()."""
    for name, module in model.named_children():
        if isinstance(module, BatchNorm2d):
            setattr(model, name, Identity())
        else:
            remove_batchnorm(module)
    return model

# ------------------------------
# main
# ------------------------------
def analyze_array_energy(
    model: torch.nn.Module,
    dataloader,
    config: Dict[str, Any],
    save_path: str = "array_energy_report.csv"
) -> Dict[float, pd.DataFrame]:
    """
    Estimate proxy array-only energy for AIHWKit-mapped model, across multiple percentiles.

    Args:
        analog_model: Inference-ready model (AIHWKit analog model)
        dataloader: DataLoader (one batch is sufficient)
        config: dict with keys:
            - Vmax: voltage max (e.g. 0.2)
            - t_read_ns: pulse width (e.g. 10.0 ns)
            - percentiles: list of percentiles to evaluate (e.g. [99.0, 100.0])
            - use_abs: use abs(activation) for voltage mapping
            - device: "cuda" or "cpu"
        save_path: CSV base path to save results (appends _p{percentile}.csv)

    Returns:
        Dict of Pandas DataFrames per percentile
    """

    device = config["device"]
    Vmax = config["Vmax"]
    t_read = config["t_read_ns"] * 1e-9
    use_abs = config["use_abs"]
    percentiles = config.get("percentiles", [100.0])
    
    # ---- remove BN before analysis ----
    model = remove_batchnorm(model)  
    analog_model = model.to(device).eval()

    # --- Step 1: Register forward hooks to capture inputs ---
    layer_inputs = {}
    hooks = []

    def make_hook(nm):
        return lambda mod, inp, out: layer_inputs.setdefault(nm, inp[0].detach())

    for name, module in analog_model.named_modules():
        if isinstance(module, (AnalogConv2d, AnalogLinear)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # --- Step 2: One forward pass to capture inputs ---
    for x, _ in dataloader:
        x = x.to(device)
        _ = analog_model(x)
        break

    for h in hooks:
        h.remove()

    # --- Step 3: Run analysis per percentile ---
    all_results = {}

    for percentile in percentiles:
        records = []
        for name, module in analog_model.named_modules():
            if name not in layer_inputs:
                continue

            x = layer_inputs[name]
            x_vals = x.abs() if use_abs else x
            scale_val = torch.quantile(x_vals.flatten(), percentile / 100.0).item()
            scale_val = scale_val if scale_val > 0 else 1.0
            V_scaled = x * (Vmax / scale_val)

            # --- NEW: Iterate over analog_tiles() instead of analog_tile ---
            G_total = None
            for tile in module.analog_tiles():
                Gs = tile.get_weights()[0]  # list of conductance tensors
                # Sum over rows for each device pair
                G_sum = sum(G.sum(dim=0) for G in Gs)
                G_total = G_sum if G_total is None else (G_total + G_sum)

            if G_total is None:
                print(f"[WARN] Layer {name} has no analog_tiles(), skipping.")
                continue

            if isinstance(module, AnalogConv2d):
                unfolded = F.unfold(
                    x, kernel_size=module.kernel_size,
                    stride=module.stride, padding=module.padding
                )
                V2 = (unfolded * (Vmax / scale_val)) ** 2
                proxy_energy = t_read * (V2 * G_total.view(-1, 1)).sum().item()
                MACs = unfolded.shape[1] * unfolded.shape[2] * module.out_channels
                zeros = (unfolded == 0).sum().item()
                total = unfolded.numel()
            elif isinstance(module, AnalogLinear):
                V2 = V_scaled ** 2
                proxy_energy = t_read * (V2 * G_total).sum().item()
                MACs = V2.shape[0] * V2.shape[1]
                zeros = (x == 0).sum().item()
                total = x.numel()
            else:
                continue

            records.append({
                "layer": name,
                "proxy_energy_J": proxy_energy,
                "MACs": MACs,
                "avg_energy_fJ_per_MAC": proxy_energy / MACs * 1e15 if MACs > 0 else np.nan,
                "zero_ratio_%": 100 * zeros / total
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df.loc["Total"] = {
                "layer": "Total",
                "proxy_energy_J": df["proxy_energy_J"].sum(),
                "MACs": df["MACs"].sum(),
                "avg_energy_fJ_per_MAC": (df["proxy_energy_J"].sum() / df["MACs"].sum()) * 1e15,
                "zero_ratio_%": np.nan
            }

        all_results[percentile] = df

    return all_results