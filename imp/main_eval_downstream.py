"""
load lottery tickets and evaluation
support datasets: cifar10, Fashionmnist, cifar100
"""

import argparse
import logging
import os
import pathlib
import random
import shutil
import time
import warnings
import copy
import math

import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

from pruning_utils import *
from dataset import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Evaluation Tickets')
############################# required settings ################################
parser.add_argument('--data', metavar='DIR', default='/home/yuanye/data',
                    help='path to dataset')
parser.add_argument('--set', type=str, default='cifar10', help='ImageNet, cifar10, cifar100, svhn')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--decreasing_lr', default='50,100', help='decreasing strategy')
parser.add_argument('--log_dir', default='runs', type=str)
parser.add_argument('--name', default='R18_inf2_s80_c10', type=str, help='experiment name')
parser.add_argument('--weight_dir', type=str, default='/home/yuanye/RST/imp/tickets/R18_inf2/weight_init.pth.tar',
                    help='path of the pretrained weight')
parser.add_argument('--mask_dir', type=str, default='/home/yuanye/RST/imp/tickets/R18_inf2/mask_state0_sp80.0.pth.tar',
                    help='path of the extracted mask')
parser.add_argument('--pytorch-pretrained', action='store_true',
                    help='If True, loads a Pytorch pretrained natural weight.')
parser.add_argument('--random', action="store_true", help="using random-init model")
parser.add_argument("--trainer", type=str, default="default", help="cs, ss, or standard training")
parser.add_argument('--attack_type', default='None', choices=['fgsm', 'fgsm-rs', 'pgd', 'free', 'None'])

############################# other settings ################################
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--adv-eval', action='store_true', default=False)
parser.add_argument('--save-model', action='store_true', help="Save the finetuned model", default=True)
parser.add_argument('--linear-eval', action='store_true', help="Linear evaluation mode", default=False)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')


def main():
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    if args.pytorch_pretrained:
        args.weight_dir = None

    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    args.ckpt_base_dir = ckpt_base_dir
    os.makedirs(args.ckpt_base_dir, exist_ok=True)

    log = logging.getLogger(__name__)
    log_path = os.path.join(run_base_dir, 'log.txt')
    handlers = [logging.FileHandler(log_path, mode='a+'),
                logging.StreamHandler()]
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=handlers)
    log.info(args)

    args.log = log
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        log.info("[INIT] Use GPU: {} for training".format(args.gpu))

    # create model with official pretrained weight or random initialization
    print("=> using model '{}', dataset '{}'".format(args.arch, args.set))
    log.info("[INIT] => using model '{}', dataset '{}'".format(args.arch, args.set))
    model, train_loader, val_loader = get_model_dataset(args)
    if_pruned = False

    writer = SummaryWriter(log_dir=log_base_dir)

    # Load pretrained weights
    if args.weight_dir:
        log.info("Loading pretrained weights")
        print("=> loading checkpoint '{}'".format(args.weight_dir))
        log.info("[LOAD] => loading checkpoint '{}'".format(args.weight_dir))
        checkpoint = torch.load(args.weight_dir)

        new_state_dict = OrderedDict()
        sd = checkpoint['state_dict']
        for k, v in sd.items():
            if 'attacker' in k:
                break
            if 'normalize' not in k:
                name = k[len('module.'):]
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        log.info("[LOAD] => Pytorch natural pretrained model")

    if args.set != 'ImageNet':
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.classes)

    print('dataparallel mode')
    log.info("[INIT] dataparallel mode")
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    # Load mask
    print("=> loading mask '{}'".format(args.mask_dir))
    log.info("[LOAD] => loading mask '{}'".format(args.mask_dir))
    mask = torch.load(args.mask_dir)
    prune_model_custom(model.module, mask['mask'], False)

    # Linear evaluation
    if args.linear_eval:
        log.info("[INIT] => Linear Evaluation Mode")
        freeze_model(log, model.module, 4)

    # Get trainer and tester
    train, validate, validate_adv, modifier = get_trainer(args)

    cudnn.benchmark = True

    all_result = {}
    all_result['train'] = []
    all_result['nat_acc'] = []
    all_result['adv_acc'] = []
    all_result['robustness_at_best_acc'] = 0.0
    all_result['best_acc1'] = 0.0
    all_result['best_epoch'] = 0

    start_epoch = 0
    mask_sp = check_sparsity(model, use_mask=True, conv1=False)
    log.info(f"[EVAL] Mask Sparsity: {mask_sp:.2f}")
    all_result['sparsity'] = mask_sp

    for epoch in range(start_epoch, args.epochs):
        log.info("-" * 150)
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        log.info(
            f"[TRAIN] Epoch {epoch} start, lr {optimizer.state_dict()['param_groups'][0]['lr']}")

        # train for one epoch
        train_acc1, train_acc5 = train(train_loader, model, criterion, optimizer, epoch, args, writer)
        log.info("[TRAIN] Train done! Train@1: %.2f, Train@5: %.2f", train_acc1, train_acc5)
        all_result['train'].append(train_acc1)

        # evaluate on validation set
        nat_acc1, nat_acc5 = validate(val_loader, model, criterion, args, writer, epoch)
        log.info("[EVAL] Natural eval done! Nat@1: %.2f, Nat@5: %.2f", nat_acc1, nat_acc5)
        all_result['nat_acc'].append(nat_acc1)

        if args.adv_eval:
            # evaluate on adversary validation set
            adv_acc1, adv_acc5 = validate_adv(val_loader, model, criterion, args, writer, epoch)
            log.info("[EVAL] Robust eval done! Adv@1: %.2f, Adv@5: %.2f", adv_acc1, adv_acc5)
        else:
            adv_acc1, adv_acc5 = -1, -1
        all_result['adv_acc'].append(adv_acc1)

        # remember best acc@1 and save checkpoint
        is_best = nat_acc1 > all_result['best_acc1']
        all_result['best_acc1'] = max(nat_acc1, all_result['best_acc1'])

        if is_best:
            all_result['best_epoch'] = epoch + 1
            if args.adv_eval:
                all_result['robustness_at_best_acc'] = adv_acc1
            else:
                all_result['robustness_at_best_acc'] = -1
            log.info("[EVAL] ***** This is the best epoch so far ***** ")

        log.info("[EVAL] Best Accuracy: %.2f at epoch %d, robustness at best acc %.2f", all_result['best_acc1'],
                 all_result['best_epoch'], all_result['robustness_at_best_acc'])

        scheduler.step()

        if args.save_model:
            save_checkpoint({
                'result': all_result,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': all_result['best_acc1'],
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best=is_best, checkpoint=args.ckpt_base_dir)

        else:
            save_checkpoint({
                'result': all_result
            }, is_best=False, checkpoint=args.ckpt_base_dir)

        plt.plot(all_result['train'], label='train_acc')
        plt.plot(all_result['nat_acc'], label='natural_acc')
        plt.legend()
        plt.savefig(os.path.join(log_base_dir, 'net_train.png'))
        plt.close()

        if args.adv_eval:
            plt.plot(all_result['adv_acc'], label='robust_acc')
            plt.legend()
            plt.savefig(os.path.join(log_base_dir, 'robustness.png'))
            plt.close()

    check_sparsity(model, use_mask=True, conv1=False)
    log.info(all_result)


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar', best_name='model_best.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if is_best:
        bestpath = os.path.join(checkpoint, best_name)
        shutil.copyfile(filepath, bestpath)
        return bestpath

    return filepath


def get_trainer(args):
    print(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")

    if args.attack_type == 'None':
        return trainer.train, trainer.validate, trainer.validate_adv, trainer.modifier
    else:
        if args.attack_type == 'free':
            return trainer.train_adv_free, trainer.validate, trainer.validate_adv, trainer.modifier
        else:
            return trainer.train_adv, trainer.validate, trainer.validate_adv, trainer.modifier


def get_directories(args):
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{args.name}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{args.name}"
        )

    # if _run_dir_exists(run_base_dir):
    #     rep_count = 0
    #     while _run_dir_exists(run_base_dir / str(rep_count)):
    #         rep_count += 1

    #     run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)
    if not log_base_dir.exists():
        os.makedirs(log_base_dir)
    if not ckpt_base_dir.exists():
        os.makedirs(ckpt_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir


def get_model_dataset(args):
    # prepare dataset
    if args.set == 'cifar10':
        args.classes = 10
        train_loader, _, test_loader = cifar10_dataloaders(args, use_val=False)
    elif args.set == 'cifar100':
        args.classes = 100
        train_loader, _, test_loader = cifar100_dataloaders(args, use_val=False)
    elif args.set == 'svhn':
        args.classes = 10
        train_loader, _, test_loader = svhn_dataloaders(args, use_val=False)
    elif args.set == 'fmnist':
        args.classes = 10
        train_loader, _, test_loader = fashionmnist_dataloaders(args, use_val=False)
    elif args.set == 'ImageNet':
        args.classes = 1000
        train_loader, _, test_loader = imagenet_dataloaders(args, use_val=False)
    else:
        raise ValueError("Unknown Dataset")

    # prepare model
    model = models.__dict__[args.arch](pretrained=(not args.random))

    return model, train_loader, test_loader


def freeze_model(log, model, freeze_level):

    assert len([name for name, _ in list(model.named_parameters())
                if f"layer{freeze_level}" in name]), "unknown freeze level (only {1,2,3,4} for ResNets)"
    update_params = []
    freeze = True
    for name, param in model.named_parameters():
        print(name, param.size())

        if not freeze and f'layer{freeze_level}' not in name:
            log.info(f"Update {name}")
            update_params.append(param)
        else:
            log.info(f"Freeze {name}")
            param.requires_grad = False

        if freeze and f'layer{freeze_level}' in name:
            # if the freeze level is detected stop freezing onwards
            freeze = False
    return update_params


if __name__ == '__main__':
    main()
