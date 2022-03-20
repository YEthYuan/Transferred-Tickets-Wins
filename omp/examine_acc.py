import torch
import torchvision.models as models
import numpy as np

model = models.resnet18(pretrained=False)
file_path = 'runs/img-c10-resnet18-eps0-p90-finetune/checkpoint.pt.best'
state_dict = torch.load(file_path)
model.load_state_dict(state_dict['model'])
print(state_dict)

