import os
import sys
import torch

# get the parent directory and import the model
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
from model.PyTorch_CIFAR10.cifar10_models.resnet import resnet18


class ModelLoader:
    dir_name = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'model', 'pruned'))

    name_list = [
        'vanilla-Resnet18',
        'Resnet18_p0.3',
        # 'Resnet18_p0.4',
        'Resnet18_p0.5',
        'Resnet18_p0.6',
        'Resnet18_p0.7',
        # 'Resnet18_p0.8',
        'Resnet18_p0.9',
        ]


    @staticmethod
    def get_model_file(imported_model):
        if imported_model == '1':
            return 'local_pruned_model_param.pth'
        elif imported_model == '2':
            return 'FT_0.0001_50/best_model_param.pth'
        elif imported_model == '3':
            return 'test_model.pth'
        else:
            raise ValueError("Invalid model type.")


    @staticmethod
    def get_resnet18_model(pretrained=False):
        model = resnet18(pretrained=pretrained)
        return model
    
    
    @classmethod
    def load_models(cls, imported_model:int) -> dict:
        model_dict = {}
        model_file = cls.get_model_file(imported_model)
        
        for model_type in cls.name_list:
            print(f' Model Type : {model_type}')

            folder_path = os.path.join(cls.dir_name, model_type)
            model = cls.get_resnet18_model()
            
            if 'vanilla' in model_type:
                model = cls.get_resnet18_model(pretrained=True)
                print('>>> Vanilla model loaded')
                
            else:
                model = cls.get_resnet18_model(pretrained=False)
                state_dict = torch.load(os.path.join(folder_path, model_file), map_location='cpu')
                model.load_state_dict(state_dict)
                model.eval()
                print(f'>>> Model loaded from : {model_file}')
                
            # if 'vanilla' in model_type:
            #     model = resnet18(pretrained=True)
            #     print('Vanilla model loaded')
            # else:
            #     model = torch.load(os.path.join(folder_path, model_file), map_location='cpu')
            #     print(f'Model loaded from {model_file}')

            model_dict[model_type] = model

        return model_dict