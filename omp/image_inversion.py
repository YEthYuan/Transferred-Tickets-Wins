import argparse
import copy
import os
import random

import cox.store
import dill
import numpy as np
import torch as ch
from cox import utils
from robustness.tools.vis_tools import show_image_row, show_image_column
from tqdm import tqdm

from my_robustness import datasets, defaults, model_utils, train, attacker
from my_robustness.tools import helpers
from torch import nn
from torch.nn.utils import prune
from torchvision import models

from utils import constants as cs
from utils import fine_tunify, transfer_datasets

parser = argparse.ArgumentParser(description='Transfer learning via pretrained Imagenet models',
                                 conflict_handler='resolve')
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)

# Adversarial arguments
parser.add_argument('--attack-steps', type=int, help='number of steps for PGD attack', default=7)
parser.add_argument('--constraint', type=str, help='adv constraint', default='2')
parser.add_argument('--eps', type=str, help='adversarial perturbation budget', default='3')
parser.add_argument('--attack-lr', type=str, help='step size for PGD', default='10')

# Custom arguments
parser.add_argument('--dataset', type=str, default='flowers',
                    help='Dataset (Overrides the one in robustness.defaults)')
parser.add_argument('--dataset_seed', type=str, default='caltech101',
                    help='Dataset (Overrides the one in robustness.defaults)')
parser.add_argument('--data', type=str, default='/home/yuanye/data')
parser.add_argument('--out-dir', type=str, default='runs')
parser.add_argument('--exp-name', type=str, default='test-debug-run')
parser.add_argument('--arch', type=str, default='resnet18')
# parser.add_argument('--model-path', type=str, default='pretrained_models/resnet18_l2_eps3.ckpt')
parser.add_argument('--model-path', type=str, default=None)
parser.add_argument('--mask-save-dir', type=str, default=None)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--opt', type=str, default='sgd', help='choose sgd or adam')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--step-lr', type=int, default=30)
parser.add_argument('--batch-size', type=int, default=10)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--prune_rate', type=float, default=0)
parser.add_argument('--prune_percent', type=int, default=None)
parser.add_argument('--structural_prune', action='store_true',
                    help='Use the structural pruning method (currently channel pruning)')
parser.add_argument('--adv-train', type=int, default=0)
parser.add_argument('--adv-eval', type=int, default=0)
parser.add_argument('--seed', type=int, default=999)
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--conv1', action='store_true',
                    help="If true, prune the conv1, if false, skip the conv1")
parser.add_argument('--resume', action='store_true',
                    help='Whether to resume or not (Overrides the one in robustness.defaults)')
parser.add_argument('--pytorch-pretrained', action='store_true', default=True,
                    help='If True, loads a Pytorch pretrained model.')
parser.add_argument('--only-extract-mask', action='store_true',
                    help='If True, only extract the ticket from Imagenet pretrained model')
parser.add_argument('--cifar10-cifar10', action='store_true',
                    help='cifar10 to cifar10 transfer')
parser.add_argument('--shuffle_test', action='store_true', default=True)
parser.add_argument('--subset', type=int, default=None,
                    help='number of training data to use from the dataset')
parser.add_argument('--no-tqdm', type=int, default=1,
                    choices=[0, 1], help='Do not use tqdm.')
parser.add_argument('--no-replace-last-layer', action='store_true',
                    help='Whether to avoid replacing the last layer')
parser.add_argument('--freeze-level', type=int, default=-1,
                    help='Up to what layer to freeze in the pretrained model (assumes a resnet architectures)')
parser.add_argument('--additional-hidden', type=int, default=0,
                    help='How many hidden layers to add on top of pretrained network + classification layer')
parser.add_argument('--per-class-accuracy', action='store_true', help='Report the per-class accuracy. '
                                                                      'Can be used only with pets, caltech101, caltech256, aircraft, and flowers.')


# Custom loss for inversion
def inversion_loss(model, inp, targ):
    _, rep = model(inp, with_latent=True, fake_relu=True)
    loss = ch.div(ch.norm(rep - targ, dim=1), ch.norm(targ, dim=1))
    return loss, None


DATA_PATH_DICT = {
    'CIFAR': '/home/yuanye/data/cifar10',
    'RestrictedImageNet': '/scratch/engstrom_scratch/imagenet',
    'ImageNet': '/scratch/engstrom_scratch/imagenet',
    'H2Z': '/scratch/datasets/A2B/horse2zebra',
    'A2O': '/scratch/datasets/A2B/apple2orange',
    'S2W': '/scratch/datasets/A2B/summer2winter_yosemite'
}
# PGD parameters
kwargs = {
    'custom_loss': inversion_loss,
    'constraint': '2',
    'eps': 10000,
    'step_size': 1,
    'iterations': 10000,
    'do_tqdm': True,
    'targeted': True,
    'use_best': False
}
# Constants
NOISE_SCALE = 20

DATA_SHAPE = 224  # Image size (fixed for dataset)
# DATA_SHAPE = 32 if DATA == 'CIFAR' else 224 # Image size (fixed for dataset)
REPRESENTATION_SIZE = 2048  # Size of representation vector (fixed for model)


def main(args, store):
    if args.seed is not None:
        random.seed(args.seed)
        ch.manual_seed(args.seed)
        ch.cuda.manual_seed(args.seed)
        ch.cuda.manual_seed_all(args.seed)
    '''Given arguments and a cox store, trains as a model. Check out the 
    argparse object in this file for argument options.
    '''
    if args.prune_percent:
        args.prune_rate = args.prune_percent / 100
        print("current prune_rate=", args.prune_rate)

    if args.pytorch_pretrained:
        args.model_path = None

    ds, train_loader, validation_loader = get_dataset_and_loaders(args, args.dataset, batch_size=1)
    data_iterator = enumerate(validation_loader)
    ds_seed, train_loader_seed, validation_loader_seed = get_dataset_and_loaders(args, args.dataset_seed,
                                                                                 batch_size=args.batch_size)
    data_iterator_seed = enumerate(validation_loader_seed)

    model, checkpoint = get_model(args, ds)
    check_sparsity(model, use_mask=False, conv1=args.conv1)

    if args.structural_prune:
        cur_prune_rate = args.prune_rate
        print("L2 Structured Pruning (Conv2d channel pruning) Start")
        prune_model_structural(model, cur_prune_rate, conv1=args.conv1)
        print("L2 Structured Pruning (Conv2d channel pruning) Done!")
    else:
        cur_prune_rate = args.prune_rate
        print("L1 Unstructured Pruning Start")
        pruning_model(model, cur_prune_rate, conv1=args.conv1)
        print("L1 Unstructured Pruning Done!")

    # Extract mask
    check_sparsity(model, use_mask=True, conv1=args.conv1)
    current_mask = extract_mask(model.state_dict())
    remove_prune(model, conv1=args.conv1)

    model.eval()

    for i in range(19):
        _, (im, targ) = next(data_iterator)  # Images to invert
    show_image_column([im.cpu()], [r"Target ($x_2$)"], fontsize=22, filename="svg/target.svg")
    for i in range(1):
        _, (im_n, targ) = next(data_iterator_seed)
    im = im.repeat(args.batch_size, 1, 1, 1)
    show_image_row([im_n.cpu()], [r"Source ($x_1$)"],
                fontsize=22, filename="svg/source.svg")
    # show_image_column([im.cpu()], [r"Target ($x_2$)"], fontsize=22, filename=f"{args.prune_rate}_target.svg")

    with ch.no_grad():
        (_, rep), _ = model(im.cuda(), with_latent=True)  # Corresponding representation

    # im_n = ch.randn_like(im) / NOISE_SCALE + 0.5  # Seed for inversion (x_0)

    _, xadv = model(im_n.cuda(), rep.clone(), make_adv=True, **kwargs)  # Image inversion using PGD

    show_image_row([xadv.detach().cpu()],
                   [f"Standard {1-args.prune_rate:.1f}"],
                   fontsize=22, filename=f"svg/{args.prune_rate}_nat_result.svg")


def get_per_class_accuracy(args, loader):
    '''Returns the custom per_class_accuracy function. When using this custom function         
    look at only the validation accuracy. Ignore trainig set accuracy.
    '''

    def _get_class_weights(args, loader):
        '''Returns the distribution of classes in a given dataset.
        '''
        # if args.dataset in ['pets', 'flowers']:
        #     targets = loader.dataset.targets
        #
        # elif args.dataset in ['caltech101', 'caltech256']:
        #     targets = np.array([loader.dataset.ds.dataset.y[idx]
        #                         for idx in loader.dataset.ds.indices])
        #
        # elif args.dataset == 'aircraft':
        #     targets = [s[1] for s in loader.dataset.samples]
        targets = []
        targets = np.array(targets)
        print('Calculating the class weights ... ... ')
        for _, target in tqdm(loader):
            targets = np.append(targets, target.numpy())

        counts = np.unique(targets, return_counts=True)[1]
        class_weights = counts.sum() / (counts * len(counts))
        print("class weight: ", class_weights)
        return ch.Tensor(class_weights)

    class_weights = _get_class_weights(args, loader)

    def custom_acc(logits, labels):
        '''Returns the top1 accuracy, weighted by the class distribution.
        This is important when evaluating an unbalanced dataset. 
        '''
        batch_size = labels.size(0)
        maxk = min(5, logits.shape[-1])
        prec1, _ = helpers.accuracy(
            logits, labels, topk=(1, maxk), exact=True)

        normal_prec1 = prec1.sum(0, keepdim=True).mul_(100 / batch_size)
        weighted_prec1 = prec1 * class_weights[labels.cpu()].cuda()
        weighted_prec1 = weighted_prec1.sum(
            0, keepdim=True).mul_(100 / batch_size)

        return weighted_prec1.item(), normal_prec1.item()

    return custom_acc


def get_dataset(args):
    ds = None
    if args.dataset in ['imagenet', 'stylized_imagenet']:
        ds = datasets.ImageNet(args.data)
    else:
        pass

    return ds


def get_dataset_and_loaders(args, dataset, batch_size):
    '''Given arguments, returns a datasets object and the train and validation loaders.
    '''
    if dataset in ['pets', 'caltech101', 'caltech256', 'flowers', 'aircraft']:
        args.per_class_accuracy = True
    else:
        args.per_class_accuracy = False
    if dataset in ['imagenet', 'stylized_imagenet']:
        ds = datasets.ImageNet(args.data)
        train_loader, validation_loader = ds.make_loaders(
            only_val=args.eval_only, batch_size=batch_size, workers=args.workers)
    elif args.cifar10_cifar10:
        ds = datasets.CIFAR('/tmp')
        train_loader, validation_loader = ds.make_loaders(
            only_val=args.eval_only, batch_size=batch_size, workers=args.workers)
    else:
        ds, (train_loader, validation_loader) = transfer_datasets.make_loaders(args=args,
                                                                               ds=dataset, batch_size=batch_size,
                                                                               workers=args.workers, subset=args.subset)
        if type(ds) == int:
            new_ds = datasets.CIFAR("/tmp")
            new_ds.num_classes = ds
            new_ds.mean = ch.tensor([0., 0., 0.])
            new_ds.std = ch.tensor([1., 1., 1.])
            ds = new_ds
    return ds, train_loader, validation_loader


def resume_finetuning_from_checkpoint(args, ds, finetuned_model_path):
    '''Given arguments, dataset object and a finetuned model_path, returns a model
    with loaded weights and returns the checkpoint necessary for resuming training.
    '''
    print('[Resuming finetuning from a checkpoint...]')
    if args.dataset in list(transfer_datasets.DS_TO_FUNC.keys()) and not args.cifar10_cifar10:
        model, _ = model_utils.make_and_restore_model(
            arch=pytorch_models[args.arch](
                args.pytorch_pretrained) if args.arch in pytorch_models.keys() else args.arch,
            dataset=datasets.ImageNet(''), add_custom_forward=args.arch in pytorch_models.keys())
        while hasattr(model, 'model'):
            model = model.model
        model = fine_tunify.ft(
            args.arch, model, ds.num_classes, args.additional_hidden)
        model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=ds, resume_path=finetuned_model_path,
                                                               add_custom_forward=args.additional_hidden > 0 or args.arch in pytorch_models.keys())
    else:
        model, checkpoint = model_utils.make_and_restore_model(
            arch=args.arch, dataset=ds, resume_path=finetuned_model_path)
    return model, checkpoint


def get_model(args, ds):
    '''Given arguments and a dataset object, returns an ImageNet model (with appropriate last layer changes to 
    fit the target dataset) and a checkpoint.The checkpoint is set to None if noe resuming training.
    '''
    finetuned_model_path = os.path.join(
        args.out_dir, args.exp_name, 'checkpoint.pt.latest')
    if args.resume and os.path.isfile(finetuned_model_path):
        model, checkpoint = resume_finetuning_from_checkpoint(
            args, ds, finetuned_model_path)
    else:
        if args.dataset in list(transfer_datasets.DS_TO_FUNC.keys()) and not args.cifar10_cifar10:
            model, _ = model_utils.make_and_restore_model(
                arch=pytorch_models[args.arch](
                    args.pytorch_pretrained) if args.arch in pytorch_models.keys() else args.arch,
                dataset=datasets.ImageNet(''), resume_path=args.model_path, pytorch_pretrained=args.pytorch_pretrained,
                add_custom_forward=args.arch in pytorch_models.keys())
            checkpoint = None
        else:
            model, _ = model_utils.make_and_restore_model(arch=args.arch, dataset=ds,
                                                          resume_path=args.model_path,
                                                          pytorch_pretrained=args.pytorch_pretrained)
            checkpoint = None

        if not args.no_replace_last_layer and not args.eval_only:
            print(f'[Replacing the last layer with {args.additional_hidden} '
                  f'hidden layers and 1 classification layer that fits the {args.dataset} dataset.]')
            while hasattr(model, 'model'):
                model = model.model
            model = fine_tunify.ft(
                args.arch, model, ds.num_classes, args.additional_hidden)
            model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=ds,
                                                                   add_custom_forward=args.additional_hidden > 0 or args.arch in pytorch_models.keys())
        else:
            print('[NOT replacing the last layer]')
    return model, checkpoint


def freeze_model(model, freeze_level):
    '''
    Freezes up to args.freeze_level layers of the model (assumes a resnet model)
    '''
    # Freeze layers according to args.freeze-level
    update_params = None
    if freeze_level != -1:
        # assumes a resnet architecture
        assert len([name for name, _ in list(model.named_parameters())
                    if f"layer{freeze_level}" in name]), "unknown freeze level (only {1,2,3,4} for ResNets)"
        update_params = []
        freeze = True
        for name, param in model.named_parameters():
            print(name, param.size())

            if not freeze and f'layer{freeze_level}' not in name:
                print(f"[Appending the params of {name} to the update list]")
                update_params.append(param)
            else:
                param.requires_grad = False

            if freeze and f'layer{freeze_level}' in name:
                # if the freeze level is detected stop freezing onwards
                freeze = False
    return update_params


def args_preprocess(args):
    '''
    Fill the args object with reasonable defaults, and also perform a sanity check to make sure no
    args are missing.
    '''
    if args.adv_train and eval(args.eps) == 0:
        print('[Switching to standard training since eps = 0]')
        args.adv_train = 0

    if args.pytorch_pretrained:
        assert not args.model_path, 'You can either specify pytorch_pretrained or model_path, not together.'

    # CIFAR10 to CIFAR10 assertions
    if args.cifar10_cifar10:
        assert args.dataset == 'cifar10'

    if args.data != '':
        cs.CALTECH101_PATH = cs.CALTECH256_PATH = cs.PETS_PATH = cs.CARS_PATH = args.data
        cs.FGVC_PATH = cs.FLOWERS_PATH = cs.DTD_PATH = cs.SUN_PATH = cs.FOOD_PATH = cs.BIRDS_PATH = args.data

    ALL_DS = list(transfer_datasets.DS_TO_FUNC.keys()) + \
             ['imagenet', 'breeds_living_9', 'stylized_imagenet']
    assert args.dataset in ALL_DS

    # Important for automatic job retries on the cluster in case of premptions. Avoid uuids.
    assert args.exp_name != None

    # Preprocess args
    args = defaults.check_and_fill_args(args, defaults.CONFIG_ARGS, None)
    if not args.eval_only:
        args = defaults.check_and_fill_args(args, defaults.TRAINING_ARGS, None)
    if args.adv_train or args.adv_eval:
        args = defaults.check_and_fill_args(args, defaults.PGD_ARGS, None)
    args = defaults.check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, None)

    return args


def check_sparsity(model, use_mask=True, conv1=True):
    sum_list = 0
    zero_sum = 0

    for module_name, module in model.named_modules():
        if isinstance(module, ch.nn.Conv2d):
            if 'conv1' in module_name and 'layer' not in module_name:
                if conv1:
                    module_sum_list, module_zero_sum = check_module_sparsity(module, use_mask=use_mask)
                    sum_list += module_sum_list
                    zero_sum += module_zero_sum
                else:
                    print('skip conv1 for sparsity checking')
            else:
                module_sum_list, module_zero_sum = check_module_sparsity(module, use_mask=use_mask)
                sum_list += module_sum_list
                zero_sum += module_zero_sum

    if zero_sum:
        remain_weight_rate = 100 * (1 - zero_sum / sum_list)
        print('* remain weight ratio = ', 100 * (1 - zero_sum / sum_list), '%')
    else:
        print('no weight for calculating sparsity')
        remain_weight_rate = None

    return remain_weight_rate


def check_module_sparsity(module, use_mask=True):
    sum_list = 0
    zero_sum = 0

    if use_mask:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name:
                sum_list += buffer.nelement()
                zero_sum += ch.sum(buffer == 0).item()

    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name:
                sum_list += param.nelement()
                zero_sum += ch.sum(param == 0).item()

    return sum_list, zero_sum


def pruning_model(model, px, conv1=False):
    parameters_to_prune = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if 'conv1' in name and 'layer' not in name:
                if conv1:
                    parameters_to_prune.append((m, 'weight'))
                else:
                    print('skip conv1 for L1 unstructure global pruning')
            else:
                parameters_to_prune.append((m, 'weight'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )


def prune_model_structural(model, cur_prune_rate, conv1=False):
    print('Apply Structured L2 Channel Pruning (all conv layers)')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if 'conv1' in name and 'layer' not in name:
                if conv1:
                    prune.ln_structured(m, name="weight", amount=cur_prune_rate, n=2, dim=0)
                else:
                    print('skip conv1 for Structured L2 Channel Pruning')
            else:
                prune.ln_structured(m, name="weight", amount=cur_prune_rate, n=2, dim=0)


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


def extract_mask(model_dict):
    new_dict = {}
    for key in model_dict.keys():
        if 'mask' in key:
            new_mask_key = 'module.' + key
            new_dict[new_mask_key] = copy.deepcopy(model_dict[key])

    return new_dict


if __name__ == "__main__":
    args = parser.parse_args()
    args = args_preprocess(args)

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

    # Create store and log the args
    store = cox.store.Store(args.out_dir, args.exp_name)
    if 'metadata' not in store.keys:
        args_dict = args.__dict__
        schema = cox.store.schema_from_dict(args_dict)
        store.add_table('metadata', schema)
        store['metadata'].append_row(args_dict)
    else:
        print('[Found existing metadata in store. Skipping this part.]')
    main(args, store)
