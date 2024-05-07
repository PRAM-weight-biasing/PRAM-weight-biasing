import os
from datetime import datetime


def MakeFolder(x) -> str:  # create new directory
    now = datetime.now()
    runtime = now.strftime('%Y-%m-%d %H:%M:%S')
    folder_name = "Test_"+ runtime
    dir_name = os.getcwd() + '/TestRun/'

    folder_path = dir_name + folder_name
    os.makedirs(folder_path, exist_ok=True)

    return folder_path