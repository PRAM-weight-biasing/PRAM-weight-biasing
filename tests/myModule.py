import os
from datetime import datetime
from typing import Optional
import random
import torch
import numpy as np


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
    
def seed_everything(seed: Optional[int] = 42) -> None:
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