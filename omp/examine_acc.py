import torch
import torchvision.models as models
import numpy as np

model = models.resnet18(pretrained=False)
file_path = '/home/yuanye/RST/omp/extracted_masks/adv_pr0.9_ticket.pth'
state_dict = torch.load(file_path)
model.load_state_dict(state_dict['model'])
print(state_dict)

