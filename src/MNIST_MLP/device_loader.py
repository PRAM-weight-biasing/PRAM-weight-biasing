import torch

"""==================================="""

class DeviceDataLoader():
    """class for loading model or dataset to targetted device
    """
    def __init__(self, dl, device) -> None:
        self.dl     = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl:
            yield self.to_device(b)
            
    def __len__(self):
        return len(self.dl)
    
    def to_device(self, data):
        """sends data to device specified in object

        Args:
            data (model or dataset): dataset (neural net model or training dataset)

        Returns:
            _type_: wrapped dataset (do not assign model to a variable)
        """
        self.__check_device()
        if isinstance(data, (list, tuple)):
            return [self.to_device(x) for x in data]
        return data.to(self.device, non_blocking=True)
    
    def __check_device(self) -> None:
        """internal function for checking validity of device string

        Args:
            device (str): device name string
        """
        if self.device != 'cpu' or self.device != 'cuda': # default case
            self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        return None
    
def to_device(data, device: str):
    """sends data to target device

    Args:
        data (model or data): model or data to transfer
        device (str): device name string

    Returns:
        _type_: sent data
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x) for x in data]
    return data.to(device, non_blocking=True)

def check_device(device: str) -> str:
    """Check validity of device string

    Args:
        device (str): device to train or inference

    Returns:
        str: valid device to train or inference
    """
    if device != 'cpu' or device != 'cuda': # default case
            device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return device
