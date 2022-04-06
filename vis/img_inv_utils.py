import os

import dill
import torch
from robustness import datasets
from robustness.attacker import AttackerModel
from robustness.model_utils import DummyModel
from robustness.tools.custom_modules import SequentialWithArgs
from torch import nn
from torch.nn.utils import prune
from torchvision import models

pytorch_models = {
    'alexnet': models.alexnet,
    'vgg16': models.vgg16,
    'vgg16_bn': models.vgg16_bn,
    'squeezenet': models.squeezenet1_0,
    'densenet': models.densenet161,
    'shufflenet': models.shufflenet_v2_x1_0,
    'mobilenet': models.mobilenet_v2,
    'resnext50_32x4d': models.resnext50_32x4d,
    'mnasnet': models.mnasnet1_0,
}


def prune_model_custom(model, mask_dict, conv1=False):
    print('start unstructured pruning with custom mask')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if 'conv1' in name and 'layer' not in name:
                if conv1:
                    prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name + '.weight_mask'])
                else:
                    print('skip conv1 for custom pruning')
            else:
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name + '.weight_mask'])


def remove_prune(model, conv1=False):
    print('remove pruning')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if 'conv1' in name and 'layer' not in name:
                if conv1:
                    prune.remove(m, 'weight')
                else:
                    print('skip conv1 for remove pruning')
            else:
                prune.remove(m, 'weight')


def get_model(ds, arch, pretrained_path=None, additional_hidden=0):

    model, _ = make_and_restore_model(
        arch=arch, dataset=datasets.ImageNet(''), resume_path=pretrained_path, pytorch_pretrained=True if pretrained_path is None else False,
        add_custom_forward=arch in pytorch_models.keys())
    checkpoint = None

    print(f'[Replacing the last layer with {additional_hidden} '
          f'hidden layers and 1 classification layer that fits the new dataset.]')
    while hasattr(model, 'model'):
        model = model.model
    model = ft(arch, model, ds.num_classes, additional_hidden)
    model, checkpoint = make_and_restore_model(arch=model, dataset=ds, add_custom_forward=arch in pytorch_models.keys())

    return model, checkpoint


def ft(model_name, model_ft, num_classes, additional_hidden=0):
    if model_name in ["resnet", "resnet18", "resnet50", "wide_resnet50_2", "wide_resnet50_4", "resnext50_32x4d", 'shufflenet']:
        num_ftrs = model_ft.fc.in_features
        # The two cases are split just to allow loading
        # models trained prior to adding the additional_hidden argument
        # without errors
        if additional_hidden == 0:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        else:
            model_ft.fc = SequentialWithArgs(
                *list(sum([[nn.Linear(num_ftrs, num_ftrs), nn.ReLU()] for i in range(additional_hidden)], [])),
                nn.Linear(num_ftrs, num_classes)
            )
        input_size = 224
    elif model_name == "alexnet":
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    elif "vgg" in model_name:
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    elif model_name == "squeezenet":
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224
    elif model_name == "densenet":
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name in ["mnasnet", "mobilenet"]:
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        raise ValueError("Invalid model type, exiting...")

    return model_ft


def make_and_restore_model(*_, arch, dataset, resume_path=None,
                           parallel=False, pytorch_pretrained=False, add_custom_forward=False):
    if (not isinstance(arch, str)) and add_custom_forward:
        arch = DummyModel(arch)

    classifier_model = dataset.get_model(arch, pytorch_pretrained) if \
        isinstance(arch, str) else arch

    model = AttackerModel(classifier_model, dataset)

    # optionally resume from a checkpoint
    checkpoint = None
    if resume_path and os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path, pickle_module=dill)

        # Makes us able to load models saved with legacy versions
        state_dict_path = 'model'
        if not ('model' in checkpoint):
            state_dict_path = 'state_dict'

        sd = checkpoint[state_dict_path]
        sd = {k[len('module.'):]: v for k, v in sd.items()}
        model.load_state_dict(sd)
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
    elif resume_path:
        error_msg = "=> no checkpoint found at '{}'".format(resume_path)
        raise ValueError(error_msg)

    if parallel:
        model = torch.nn.DataParallel(model)
    model = model.cuda()

    return model, checkpoint