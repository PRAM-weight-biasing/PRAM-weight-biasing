# -*- coding: utf-8 -*-

""" calculate the power consumption of each model """

# import customized files
import os
from pathlib import Path
import module.myModule as myModule
from module.model_loader import ModelLoader 
from module.get_power import analyze_array_energy


# Setting
myModule.start_timer()
myModule.fix_seed()

# load the model
imported_model = '2'  # input("Model type? (1: Pruned /2: FineTuned / 3: Test) : ")
model_dict = ModelLoader.load_models(imported_model)

# set dataloader
_, testloader = myModule.set_dataloader(data_type="cifar10")

# output dir & base name
save_dir = Path("../results")
save_dir.mkdir(parents=True, exist_ok=True)
base_name = "array_energy_report"

config = {
    "device": "cuda",
    "Vmax": 0.2,
    "t_read_ns": 10.0,
    "percentiles": [100.0], # [99.0, 99.5, 100.0],
    "use_abs": True,

    # --- converter 선택 & 파라미터 ---
    "converter": "single",
    # "converter": "mapped",         # or "mapped_1yr"
    "g_max": 25.0,
    "g_min": 0.1,
    # "distortion_f": 0.0,
    }

for model_name, model in model_dict.items():
    print(f"\n[>] Analyzing model: {model_name}")
    
    # analyze
    results = analyze_array_energy(
        model=model,
        dataloader=testloader,
        config = config,
        save_path=f"{base_name}.csv",
        )
    
    # save results per percentile
    for pct, df in results.items():
        df.insert(0, "model", model_name)  # add model name column
        df["converter"] = config["converter"]
        df["distortion_f"] = config["distortion_f"]

        pct_str = f"p{str(pct).replace('.', '_')}"
        file_path = save_dir / f"{base_name}_{pct_str}.csv"

        write_header = not os.path.exists(file_path)
        df.to_csv(file_path, mode='a', header=write_header, index=False)
        print(f"    └─ saved: {file_path.name} (append: {not write_header})")

myModule.end_timer()