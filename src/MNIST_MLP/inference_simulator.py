import torch

import aihwkit
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.devices import OneSidedUnitCell

"""========================================================================="""

class inference_model():
    def __init__(self, folder_path: str) -> None:
        """convert software (torch) trained model to aihwkit model for
            inference accuracy over time evaluation

        Args:
            folder_path (str): target folder to evaluate. Select main folder,
                                not individual network folder

        Returns:
            _type_: None
        """
        self.folder_path    = folder_path
        self.list_of_models = []
        
        self.__getModel() # load 10 models into list
        return None
    
    def __getModel(self) -> None:
        """private function for loading 10 different networks of target to 
        object attribute "list_of_models"

        Returns:
            _type_: _description_
        """
        for i in range(10):
            # total 10 different random seeds
            temp_path   = self.folder_path + '/' + f"seed{(i+1)*100}/best_model.pth"
            self.list_of_models.append(torch.load(temp_path))

        return None

    def setConfig(self):
        # initialize resistive processing unit configuration
        rpu_config          = InferenceRPUConfig()
        rpu_config.device   = OneSidedUnitCell()

        pass

    def convertModel(self):
        pass

    def evalHWModel(self):
        pass

