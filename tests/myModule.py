import os
from datetime import datetime
from typing import Optional
import random
import torch
import numpy as np
import gc
import pandas as pd
import sys


def MakeFolder(test: Optional[str]) :
    """create the new directory for test

    Args:
        test (Optional[str]): description of test to use as a file name

    Returns:
        str : new folder path
    """
    
    now = datetime.now()
    runtime = now.strftime('%Y-%m-%d_%H-%M')
    folder_name = "Test_"+ runtime
    dir_name = os.getcwd() + '/TestRun/'

    test = None if test is None else test
    
    folder_path = dir_name + folder_name + test
    os.makedirs(folder_path, exist_ok=True)

    return folder_path


def get_unique_filename(directory:str, base_name:str, extension:str) -> str:
    """
    Given a directory, base file name, and extension, returns a unique file name.
    If a file with the same name already exists, appends an incrementing suffix.

    Args:
        directory (str): a folder path
        base_name (str): a base name of the file
        extension (str): a type of the file

    Returns:
        str: new file name with a numeric suffix
    """
    
    # Construct the initial file name with the given base name and extension
    file_name = f"{base_name}.{extension}"
    file_path = os.path.join(directory, file_name)

    # If the file doesn't exist, return the original name
    if not os.path.exists(file_path):
        return file_name
    else:
        # Otherwise, find a unique name by appending a numeric suffix
        suffix = 1
        while os.path.exists(file_path):
            # Generate a new file name with a suffix
            new_file_name = f"{base_name}_({suffix}).{extension}"
            new_file_path = os.path.join(directory, new_file_name)
            file_path = new_file_path
            suffix += 1

        return new_file_name
    
def fix_seed(seed: Optional[int] = 42) -> None:
    """fix the random seed number

    Args:
        seed (Optional[int]): _the random seed number_. Defaults to 42.
    """
    
    random.seed(seed)   
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def clear_memory():
    """
    clear all the memory
    """
    
    # for cpu
    gc.collect()
    
    # for gpu (pytorch)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
 
class trace():        
    def __init__(self):
        self.trace_data = pd.DataFrame(columns=["File", "Function", "Line"])
        
    def start_trace(self):
        sys.settrace(self.trace_callback)
        
    def trace_callback(self, frame, event, arg):
        """ trace the used modules with directory

        Args:
            frame (_type_): currently running frame
            event (_type_): type of events (e.g., "call", "line")
            arg (_type_): additional information

        Returns:
            _type_: function after tracing
        """
        folder_path = "/home/syoon/CM/compact_model_nmdl/.venv/lib/python3.10/site-packages/aihwkit"
        
        if event == "call":  # 함수 호출 시점만 추적
            file_name = frame.f_code.co_filename
            if folder_path in file_name:  # .venv/aihwkit 폴더 내부만 추적
                func_name = frame.f_code.co_name
                line_number = frame.f_lineno
                # print(f"Function '{func_name}' called in {file_name}")
                
                self.trace_data = pd.concat([self.trace_data, pd.DataFrame([{
                "File": file_name,
                "Function": func_name,
                "Line": line_number
                }])], ignore_index=True)
                
        return self.trace_callback
    
    def save_trace_results(self):
        """ save results to a csv file
        """
        sys.settrace(None)
        self.trace_data.to_csv("trace_results.csv", index=False, encoding="utf-8")