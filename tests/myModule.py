import os
from datetime import datetime
from typing import Optional
import random
import torch
import numpy as np
import gc
import pandas as pd
import sys
import time
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms


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
        

def set_dataloader(data_type: str):
    """set test dataloader

    Args:
        data_type (str): MNIST / CIFAR-10

    Returns:
        train dataloader (DataLoader)
        test dataloader (DataLoader)
    """
    if data_type == 'mnist':
        
        # set transform
        mnist_transform = transforms.Compose([
                    transforms.ToTensor(),   # transform : convert image to tensor. Normalized to 0~1
                    transforms.Normalize((0.1307,), (0.3081,))
        ])

        # mnist dataset
        mnist_train = dsets.MNIST(root='dataset/',
                          train=True,
                          transform=mnist_transform,  
                          download=True)
        
        mnist_test = dsets.MNIST(root='dataset/',
                                train=False,
                                transform=mnist_transform,
                                download=True)

        batch_size = 100
        
        trainloader = DataLoader(mnist_train, batch_size= batch_size, shuffle=True, num_workers=2)
        testloader = DataLoader(mnist_test, batch_size= batch_size, shuffle=False, num_workers=2)

    elif data_type == 'cifar10':
        
        # set transform
        cifar10_transform_train = transforms.Compose([
                                transforms.RandomCrop(32, padding=4),  # for data augmentation
                                transforms.RandomHorizontalFlip(),     # for data augmentation
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        ])
        
        cifar10_transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        ])

        # cifar10 dataset
        cifar10_train = dsets.CIFAR10(root='dataset/',
                                train=True,
                                download=True,
                                transform=cifar10_transform_train)
        
        cifar10_test = dsets.CIFAR10(root='dataset/',
                                train=False,
                                download=True,
                                transform=cifar10_transform_test)

        batch_size=200
        
        trainloader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=2) 
        testloader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=2) 
    
    return trainloader, testloader
 
def start_timer():
    global start_time
    start_time = time.time()
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[Start] Script execution started at {start_time_str}")

def end_timer():
    sec = time.time() - start_time
    times = str(datetime.timedelta(seconds=sec))
    short = times.split(".")[0]  # until sec.
    end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[End] Script execution finished at {end_time_str}, runtime: {short} sec\n")

 
 
class trace():   
    """ trace the imported files while running code """     
    
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