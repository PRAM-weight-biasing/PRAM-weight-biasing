import os
from time import strftime, localtime
import pandas as pd

"""================================="""

def sort_folders(param_list: list, batch_size: int, remarks: str, base_path: str) -> list:
    """Creates folders and returns "correct" hyperparameter list of list

    Args:
        param_list (list): hyperparameter list of list
        remarks (str): remarks on this run
        base_path (str): base directory (ex. "./results/MNIST_MLP")

    Returns:
        list: real hyperparameter list
    """
    # folder naming rule is (date) + (remarks)
    main_folder_name    = strftime('%Y%m%d', localtime()) + '-' + remarks[0:30]
    main_folder_path    = base_path+'/'+main_folder_name
    while True:
        if os.path.exists(main_folder_path):
            main_folder_path = main_folder_path + "_"
        else:
            break
    os.mkdir(main_folder_path) 
    
    dummy_counter   = 1
    for sub_list in param_list:
        param_naming        = sub_list[2]
        
        if param_naming.upper() == "AUTO":
            param_naming    = 'EXP' + str(dummy_counter)
        temp_folder_path    = main_folder_path + '/' + param_naming
        dummy_counter       += 1
        os.mkdir(temp_folder_path)
        sub_list[2]         = temp_folder_path
        for i in range(10):
            seed_folder_path    = temp_folder_path + f'/seed{(i+1) * 100}'
            os.mkdir(seed_folder_path)
        
    # save split test summary as csv under main folder
    df = pd.DataFrame(param_list, columns=['LearningRate','Epochs','Directory'])
    df.to_csv(main_folder_path+f'/summary(batchsize{batch_size}).csv')
    
    return param_list