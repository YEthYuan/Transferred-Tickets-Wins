import argparse
import copy
import os

import cox.store
import dill
import numpy as np
import torch as ch
from cox import utils
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
parser.add_argument('--dataset', type=str, default='cifar10',
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
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--prune_rate', type=float, default=0.9)
parser.add_argument('--prune_percent', type=int, default=None)
parser.add_argument('--structural_prune', action='store_true',
                    help='Use the structural pruning method (currently channel pruning)')
parser.add_argument('--adv-train', type=int, default=0)
parser.add_argument('--adv-eval', type=int, default=0)
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--resume', action='store_true',
                    help='Whether to resume or not (Overrides the one in robustness.defaults)')
parser.add_argument('--pytorch-pretrained', action='store_true',
                    help='If True, loads a Pytorch pretrained model.')
parser.add_argument('--only-extract-mask', action='store_true',
                    help='If True, only extract the ticket from Imagenet pretrained model')
parser.add_argument('--cifar10-cifar10', action='store_true',
                    help='cifar10 to cifar10 transfer')
parser.add_argument('--subset', type=int, default=None,
                    help='number of training data to use from the dataset')
parser.add_argument('--no-tqdm', type=int, default=1,
                    choices=[0, 1], help='Do not use tqdm.')
parser.add_argument('--no-replace-last-layer', action='store_true',
                    help='Whether to avoid replacing the last layer')
parser.add_argument('--freeze-level', type=int, default=4,
                    help='Up to what layer to freeze in the pretrained model (assumes a resnet architectures)')
parser.add_argument('--additional-hidden', type=int, default=0,
                    help='How many hidden layers to add on top of pretrained network + classification layer')
parser.add_argument('--per-class-accuracy', action='store_true', help='Report the per-class accuracy. '
                                                                      'Can be used only with pets, caltech101, caltech256, aircraft, and flowers.')


def main(args, store):
    '''Given arguments and a cox store, trains as a model. Check out the 
    argparse object in this file for argument options.
    '''
    if args.prune_percent:
        args.prune_rate = args.prune_percent / 100
        print("current prune_rate=", args.prune_rate)

    if args.only_extract_mask:
        args.dataset = 'imagenet'
        ds = get_dataset(args)
    else:
        ds, train_loader, validation_loader = get_dataset_and_loaders(args)

    if args.per_class_accuracy:
        assert args.dataset in ['pets', 'caltech101', 'caltech256', 'flowers', 'aircraft'], \
            f'Per-class accuracy not supported for the {args.dataset} dataset.'

        # VERY IMPORTANT
        # We report the per-class accuracy using the validation
        # set distribution. So ignore the training accuracy (as you will see it go
        # beyond 100. Don't freak out, it doesn't really capture anything),
        # just look at the validation accuarcy
        args.custom_accuracy = get_per_class_accuracy(args, validation_loader)

    model, checkpoint = get_model(args, ds)
    check_sparsity(model, use_mask=False)

    if args.structural_prune:
        cur_prune_rate = args.prune_rate
        print("L2 Structured Pruning (Conv2d channel pruning) Start")
        prune_model_structural(model, cur_prune_rate)
        print("L2 Structured Pruning (Conv2d channel pruning) Done!")
    else:
        cur_prune_rate = args.prune_rate
        print("L1 Unstructured Pruning Start")
        pruning_model(model, cur_prune_rate)
        print("L1 Unstructured Pruning Done!")

    # Extract mask
    check_sparsity(model, use_mask=True)
    current_mask = extract_mask(model.state_dict())
    remove_prune(model)

    if args.only_extract_mask:
        sd_info = {
            'model': model.state_dict(),
            'mask': current_mask,
            'prune_rate': args.prune_rate,
            'orig_model_name': args.model_path
        }
        ckpt_save_path = os.path.join(args.mask_save_dir, ("nat" if args.pytorch_pretrained else "adv") + (
            "_s" if args.structural_prune else "_uns") + f"_pr{args.prune_rate}_ticket_ImageNet.pth")
        ch.save(sd_info, ckpt_save_path)
        return

    if args.mask_save_dir:
        sd_info = {
            'model': model.state_dict(),
            'mask': current_mask,
            'prune_rate': args.prune_rate,
            'orig_model_name': args.model_path
        }
        ckpt_save_path = os.path.join(args.mask_save_dir, ("nat" if args.pytorch_pretrained else "adv") + (
            "_s" if args.structural_prune else "_uns") + f"_pr{args.prune_rate}_ticket.pth")
        ch.save(sd_info, ckpt_save_path)

    model, checkpoint = get_model(args, ds)

    if args.eval_only:
        return train.eval_model(args, model, current_mask, validation_loader, store=store)

    update_params = freeze_model(model, freeze_level=args.freeze_level)
    # update_params = None

    print(f"Dataset: {args.dataset} | Model: {args.arch}")
    best_prec = train.train_model(args, model, (train_loader, validation_loader), mask=current_mask, store=store,
                                  checkpoint=checkpoint, update_params=update_params)

    check_sparsity(model, use_mask=False)
    outp_str = ("Structural " if args.structural_prune else "Unstructural ") + (
        "nat" if args.pytorch_pretrained else "adv") + f" {args.prune_rate} best prec {best_prec} \n"
    print(outp_str)
    f = open("omp_log.txt", "a+")
    f.write(outp_str)
    f.close()


def get_per_class_accuracy(args, loader):
    '''Returns the custom per_class_accuracy function. When using this custom function         
    look at only the validation accuracy. Ignore trainig set accuracy.
    '''

    def _get_class_weights(args, loader):
        '''Returns the distribution of classes in a given dataset.
        '''
        if args.dataset in ['pets', 'flowers']:
            targets = loader.dataset.targets

        elif args.dataset in ['caltech101', 'caltech256']:
            targets = np.array([loader.dataset.ds.dataset.y[idx]
                                for idx in loader.dataset.ds.indices])

        elif args.dataset == 'aircraft':
            targets = [s[1] for s in loader.dataset.samples]

        counts = np.unique(targets, return_counts=True)[1]
        class_weights = counts.sum() / (counts * len(counts))
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


def get_dataset_and_loaders(args):
    '''Given arguments, returns a datasets object and the train and validation loaders.
    '''
    if args.dataset in ['imagenet', 'stylized_imagenet']:
        ds = datasets.ImageNet(args.data)
        train_loader, validation_loader = ds.make_loaders(
            only_val=args.eval_only, batch_size=args.batch_size, workers=0)
    elif args.cifar10_cifar10:
        ds = datasets.CIFAR('/tmp')
        train_loader, validation_loader = ds.make_loaders(
            only_val=args.eval_only, batch_size=args.batch_size, workers=0)
    else:
        ds, (train_loader, validation_loader) = transfer_datasets.make_loaders(
            ds=args.dataset, batch_size=args.batch_size, workers=0, subset=args.subset)
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


def check_sparsity(model, use_mask=True):
    sum_list = 0
    zero_sum = 0

    for module_name, module in model.named_modules():
        if isinstance(module, ch.nn.Conv2d):
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


def pruning_model(model, px):
    print('Apply Unstructured L1 Pruning Globally (all conv layers)')
    parameters_to_prune = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m, 'weight'))

    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )


def prune_model_structural(model, cur_prune_rate):
    print('Apply Structured L2 Channel Pruning (all conv layers)')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.ln_structured(m, name="weight", amount=cur_prune_rate, n=2, dim=0)


def prune_model_custom(model, mask_dict):
    print('Pruning with custom mask (all conv layers)')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            mask_name = name + '.weight_mask'
            if mask_name in mask_dict.keys():
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name + '.weight_mask'])
            else:
                print('Can not fing [{}] in mask_dict'.format(mask_name))


def remove_prune(model):
    print('Remove hooks for multiplying masks (all conv layers)')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
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
