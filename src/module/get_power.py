# -*- coding: utf-8 -*-
""" Get array-only proxy energy (software model + custom conductance converter) """
# ISSUE (25.09.09) : using soft model -> cannot handle activation results 
# (the input of the next layer is all the same regardless of methods) 

import torch
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Linear
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

# === import your converters ===
from module.g_converter import (
    SinglePairConductanceConverter,
    MappedConductanceConverter,
)

# ------------------------------
# converter registry
# ------------------------------
CONVERTER_MAP = {
    "singlepair": SinglePairConductanceConverter,
    "mapped": MappedConductanceConverter,
    "mapped_1yr": MappedConductanceConverter,
}

# ------------------------------
# helper: build converter from config
# ------------------------------
def _build_converter(cfg: Dict[str, Any]):
    """
    cfg keys (examples):
      converter: one of {"singlepair", "mapped", "mapped_1yr", ...}
      g_max, g_min, distortion_f
      f_lst (for npair/custom), g_lst (for custom)
    """
    ctype = cfg.get("converter", "singlepair").lower()
    if ctype not in CONVERTER_MAP:
        raise ValueError(f"Unknown converter type: {ctype}")
    
    converter_cls = CONVERTER_MAP[ctype]
    
    # variables
    kwargs = {
        "g_max": cfg.get("g_max", 25.0),
        "g_min": cfg.get("g_min", 0.1)
    }

    # add optional variables
    if "distortion_f" in cfg:
        kwargs["distortion_f"] = cfg["distortion_f"]

    if ctype == "mapped":
        kwargs["profile"] = "1month"
    elif ctype == "mapped_1yr":
        kwargs["profile"] = "1year"

    return converter_cls(**kwargs)

# ------------------------------
# helper: flatten conv weight to [out, in*kH*kW], linear is already [out, in]
# ------------------------------
def _weight_matrix_from_module(m: Module, device: str) -> torch.Tensor:
    if isinstance(m, Conv2d):
        return m.weight.to(device).reshape(m.weight.shape[0], -1)
    elif isinstance(m, Linear):
        return m.weight.to(device)
    else:
        raise TypeError("Only Conv2d and Linear are supported.")

# ------------------------------
# helper: column-sum of total conductance (sum over rows) -> shape [N_in_cols]
# ------------------------------
def _column_sum_G_total(gp: torch.Tensor, gm: torch.Tensor) -> torch.Tensor:
    return gp.sum(dim=0) + gm.sum(dim=0)

# ------------------------------
# main
# ------------------------------
def analyze_array_energy(
    model: Module,
    dataloader,
    config: Dict[str, Any],
    save_path: str = "array_energy_report.csv"
) -> Dict[float, pd.DataFrame]:
    
    """ Array-only proxy energy with software model + custom W->G converter. 
    Args: model: nn.Module with Conv2d / Linear layers (software model) 
    dataloader: a DataLoader (one batch is sufficient) 
    config: - device: "cuda" or "cpu" 
            - Vmax: float, e.g., 0.2 (Volts) 
            - t_read_ns: float, e.g., 10.0 (ns) 
            - percentiles: list of floats (e.g. [99.0, 99.5, 100.0]) 
            - use_abs: bool, use |activation| for scaling 
            - converter: {"singlepair","mapped","mapped_1yr","npair","custom"} (plus converter-specific params like g_max/g_min/distortion_f/f_lst/g_lst) 
    save_path: base CSV path; suffix _p{percentile}.csv will be appended. 
    
    Returns: Dict[percentile, DataFrame] 
    """
    
    # ---- unpack config ----
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    Vmax = float(config.get("Vmax", 0.2))
    t_read = float(config.get("t_read_ns", 10.0)) * 1e-9
    use_abs = bool(config.get("use_abs", True))
    percentiles: List[float] = config.get("percentiles", [99.0])

    # build converter
    converter = _build_converter(config)
    
    # ---- forward hooks to capture layer inputs (software model) ----
    all_results: Dict[float, pd.DataFrame] = {}

    for pct in percentiles:
        # prepare accumulators for this percentile
        layer_stats: Dict[str, Dict[str, float]] = {}

        def make_hook(name: str):
            def hook(mod, inp, out):
                x_in = inp[0].detach().to(device)
                x_vals = x_in.abs() if use_abs else x_in
                scale_val = torch.quantile(x_vals.flatten(), pct / 100.0).item()
                if scale_val <= 0:
                    scale_val = 1.0
                scale = Vmax / scale_val

                W = _weight_matrix_from_module(mod, device)
                conductances_list = converter.convert_to_conductances(W)[0]
                if len(conductances_list) == 2:
                    G_total_cols = _column_sum_G_total(conductances_list[0], conductances_list[1])
                else:
                    G_total_cols = torch.zeros(W.shape[1], device=W.device, dtype=W.dtype)
                    for g_plus, g_minus in zip(conductances_list[::2], conductances_list[1::2]):
                        G_total_cols += _column_sum_G_total(g_plus, g_minus)

                if isinstance(mod, Conv2d):
                    unfolded = F.unfold(
                        x_in,
                        kernel_size=mod.kernel_size,
                        stride=mod.stride,
                        padding=mod.padding,
                        dilation=mod.dilation
                    )
                    V2 = (unfolded * scale) ** 2
                    energy = (V2 * G_total_cols.view(1, -1, 1)).sum().item() * t_read
                    MACs = unfolded.shape[1] * unfolded.shape[2] * mod.out_channels
                    input_total = V2.sum().item()
                    G_total = (G_total_cols.view(1, -1, 1)).sum().item()
                    zeros = (unfolded == 0).sum().item()
                    total = unfolded.numel()
                    
                elif isinstance(mod, Linear):
                    V2 = (x_in * scale) ** 2
                    energy = (V2 * G_total_cols).sum().item() * t_read
                    MACs = V2.shape[0] * V2.shape[1]
                    G_total = G_total_cols.sum().item()
                    input_total = V2.sum().item()
                    zeros = (x_in == 0).sum().item()
                    total = x_in.numel()

                else:
                    return

                if name not in layer_stats:
                    layer_stats[name] = {"energy": 0.0, "MACs": 0, "G_total": 0, "input_total": 0, "zeros": 0, "total": 0}
                layer_stats[name]["energy"] += energy
                layer_stats[name]["MACs"] += MACs
                layer_stats[name]["G_total"] += G_total
                layer_stats[name]["input_total"] += input_total
                layer_stats[name]["zeros"] += zeros
                layer_stats[name]["total"] += total
            return hook

        # register hooks
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (Conv2d, Linear)):
                hooks.append(module.register_forward_hook(make_hook(name)))

        # run full dataloader
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(device)
                _ = model(x)

        for h in hooks:
            h.remove()

        # build results dataframe
        records = []
        for name, stats in layer_stats.items():
            MACs = stats["MACs"]
            energy = stats["energy"]
            G_total = stats["G_total"]
            input_total = stats["input_total"]
            records.append({
                "layer": name,
                "proxy_energy_J": energy,
                "MACs": MACs,
                "G_total": G_total,
                "input_total": input_total,
                "avg_energy_fJ_per_MAC": (energy / MACs) * 1e15 if MACs > 0 else np.nan,
                "zero_ratio_%": 100.0 * stats["zeros"] / stats["total"] if stats["total"] > 0 else np.nan
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df.loc["Total"] = {
                "layer": "Total",
                "proxy_energy_J": df["proxy_energy_J"].sum(),
                "MACs": df["MACs"].sum(),
                "G_total": df["G_total"].sum(),
                "input_total": df["input_total"].sum(),
                "avg_energy_fJ_per_MAC": (df["proxy_energy_J"].sum() / df["MACs"].sum()) * 1e15 if df["MACs"].sum() > 0 else np.nan,
                "zero_ratio_%": np.nan
            }

        all_results[pct] = df

    return all_results
