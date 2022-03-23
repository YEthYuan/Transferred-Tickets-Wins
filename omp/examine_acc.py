import torch
import torchvision.models as models
import numpy as np

model = models.resnet18(pretrained=False)
file_path = 'pretrained_models/imagenet_weight.pt'
state_dict = torch.load(file_path)
model.load_state_dict(state_dict['model'])
print(state_dict)

