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


def get_unique_filename(directory:str, base_name:str, extension:str) -> str:
    """
    Given a directory, base file name, and extension, returns a unique file name.
    If a file with the same name already exists, appends an incrementing suffix.
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