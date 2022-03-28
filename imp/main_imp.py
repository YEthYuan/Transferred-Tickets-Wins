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
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

from pruning_utils import *
from dataset import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
############################# required settings ################################
parser.add_argument('--data', metavar='DIR', default='/scratch/cl114/ILSVRC/Data/CLS-LOC/',
                    help='path to dataset')
parser.add_argument('--set', type=str, default='ImageNet', help='ImageNet, cifar10, cifar100, svhn')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--log_dir', default='runs', type=str)
parser.add_argument('--name', default='R18_Linf_Eps2', type=str, help='experiment name')
parser.add_argument('--model-path', type=str, default='/home/yf22/ResNet_ckpt/resnet18_linf_eps2.0.ckpt',
                    help='path of the pretrained weight')
parser.add_argument('--pytorch-pretrained', action='store_true',
                    help='If True, loads a Pytorch pretrained model.')
parser.add_argument('--percent', default=0.2, type=float, help='pruning rate for each iteration')
parser.add_argument('--states', default=19, type=int, help='number of iterative pruning states')
parser.add_argument('--start_state', default=0, type=int, help='number of iterative pruning states')
parser.add_argument('--random', action="store_true", help="using random-init model")
parser.add_argument("--trainer", type=str, default="default", help="cs, ss, or standard training")

############################# other settings ################################
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=142, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

# Adv params
parser.add_argument('--attack_type', default='None', choices=['fgsm', 'fgsm-rs', 'pgd', 'free', 'None'])
parser.add_argument('--epsilon', default=2, type=int)
parser.add_argument('--alpha', default=2.5, type=float, help='Step size')
parser.add_argument('--attack_iters', default=1, type=int, help='Attack iterations')


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
    best_acc1 = 0.0
    best_epoch = 0
    natural_acc1_at_best_robustness = 0.0

    if args.pytorch_pretrained:
        args.model_path = None

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

    # if model-path is not none, load the pretrained weight from this path
    if args.model_path:
        log.info("Loading pretrained weights")
        print("=> loading checkpoint '{}'".format(args.model_path))
        log.info("[LOAD] => loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)

        new_state_dict = OrderedDict()
        sd = checkpoint['model']
        for k, v in sd.items():
            if 'attacker' in k:
                break
            if 'normalize' not in k:
                name = k[len('module.model.'):]
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
        log.info("[LOAD] => loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
    else:
        log.info("[LOAD] => Pytorch natural pretrained model")

    if args.set != 'ImageNet':
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.classes)

    # init pretrianed weight 
    ticket_init_weight = copy.deepcopy(model.state_dict())

    print('dataparallel mode')
    log.info("[INIT] dataparallel mode")
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        log.info("resume model")
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            log.info("[LOAD] => loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            args.start_state = checkpoint['state']
            args.arch = checkpoint['arch']
            best_acc1 = checkpoint['best_acc1']
            natural_acc1_at_best_robustness = checkpoint['natural_acc1_at_best_robustness']
            if_pruned = checkpoint['if_pruned']
            ticket_init_weight = checkpoint['init_weight']

            if if_pruned:
                prune_model_custom(model.module, checkpoint['mask'], False)

            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (resume from epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            log.info("[LOAD] => loaded checkpoint '{}' (resume from epoch {})"
                     .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            log.info("[ERROR] => no checkpoint found at '{}'".format(args.resume))

    else:
        path = save_checkpoint({
            'state_dict': model.state_dict(),
        }, False, checkpoint=args.ckpt_base_dir, filename="weight_init.pth.tar")
        log.info(f"[CKPT] Initial Weight Checkpoint saved at directory: {path}")

    cudnn.benchmark = True

    train, validate, validate_adv, modifier = get_trainer(args)
    print("=> Attack type '{}'".format(args.attack_type))
    log.info("[INIT] => Attack type '{}'".format(args.attack_type))

    if args.evaluate:
        nat_acc1, nat_acc5 = validate(val_loader, model, criterion, args, writer)
        if args.attack_type is not 'None':
            adv_acc1, adv_acc5 = validate_adv(val_loader, model, criterion, args, writer)
        else:
            adv_acc1, adv_acc5 = -1, -1
        print("Evaluation result: ")
        print("Natural Acc1: %.2f, Natural Acc5: %.2f, Robust Acc1: %.2f, Robust Acc5: %.2f", nat_acc1, nat_acc5,
              adv_acc1, adv_acc5)
        log.info("Natural Acc1: %.2f, Natural Acc5: %.2f, Robust Acc1: %.2f, Robust Acc5: %.2f", nat_acc1, nat_acc5,
                 adv_acc1, adv_acc5)
        return

    for prun_iter in range(args.start_state, args.states):
        cur_sparsity = check_sparsity(model.module, use_mask=True, conv1=False)
        if cur_sparsity:
            cur_sparsity = round(cur_sparsity, 2)
        else:
            cur_sparsity = 100.00
        log.info("=" * 150)
        log.info(f"[TRAIN] State {prun_iter} start, current sparsity: {cur_sparsity}")

        if args.start_epoch == 0:
            # the best record of the current pruning state
            best_acc1 = 0.0
            best_epoch = 0
            natural_acc1_at_best_robustness = 0.0
            log.info("[INIT] Resume all records to zero")

        for epoch in range(args.start_epoch, args.epochs):
            log.info("-" * 150)
            print(optimizer.state_dict()['param_groups'][0]['lr'])
            log.info(
                f"[TRAIN] State {prun_iter} Epoch {epoch} start, sparsity {cur_sparsity}, lr {optimizer.state_dict()['param_groups'][0]['lr']}")
            # train for one epoch
            train_acc1, train_acc5 = train(train_loader, model, criterion, optimizer, epoch, args, writer)
            log.info("[TRAIN] Train done! Train@1: %.2f, Train@5: %.2f", train_acc1, train_acc5)

            # evaluate on validation set
            nat_acc1, nat_acc5 = validate(val_loader, model, criterion, args, writer, epoch)
            log.info("[EVAL] Natural eval done! Nat@1: %.2f, Nat@5: %.2f", nat_acc1, nat_acc5)

            if args.attack_type is not 'None':
                # evaluate on adversary validation set
                adv_acc1, adv_acc5 = validate_adv(val_loader, model, criterion, args, writer, epoch)
                log.info("[EVAL] Robust eval done! Adv@1: %.2f, Adv@5: %.2f", adv_acc1, adv_acc5)
            else:
                adv_acc1, adv_acc5 = -1, -1

            # remember best acc@1 and save checkpoint
            if args.attack_type is not 'None':
                is_best = adv_acc1 > best_acc1
                best_acc1 = max(adv_acc1, best_acc1)
            else:
                is_best = nat_acc1 > best_acc1
                best_acc1 = max(nat_acc1, best_acc1)

            if is_best:
                best_epoch = epoch + 1
                if args.attack_type is not 'None':
                    natural_acc1_at_best_robustness = nat_acc1
                else:
                    natural_acc1_at_best_robustness = -1
                log.info("[EVAL] ***** This is the best epoch so far ***** ")

            log.info("[EVAL] Best Accuracy: %.2f at epoch %d, Nat@1 at best robustness: %.2f", best_acc1, best_epoch,
                     natural_acc1_at_best_robustness)

            if if_pruned:
                mask_dict = extract_mask(model.state_dict())
            else:
                mask_dict = None

            path = save_checkpoint({
                'epoch': epoch + 1,
                'state': prun_iter,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'mask': mask_dict,
                'best_acc1': best_acc1,
                'natural_acc1_at_best_robustness': natural_acc1_at_best_robustness,
                'optimizer': optimizer.state_dict(),
                'if_pruned': if_pruned,
                'init_weight': ticket_init_weight
            }, is_best=False, checkpoint=args.ckpt_base_dir,
                best_name='sparsity_' + str(cur_sparsity) + '_model_best.pth.tar')
            log.info(f"[CKPT] Checkpoint saved at directory: {path}")
            # if is_best:
            #     log.info("[CKPT] This is the best record, copied to the best checkpoint. ")

            log.info(
                f"State {prun_iter} Epoch {epoch}: Adv@1: {adv_acc1}, Nat@1: {nat_acc1}, Train@1: {train_acc1}, BestEp: {best_epoch}, BestAdv@1: {best_acc1}, NAABR: {natural_acc1_at_best_robustness}")

        log.info("~" * 150)
        log.info(f"[PRUNE] State {prun_iter} pruning start! ")
        before_sp = check_sparsity(model.module, use_mask=True, conv1=False)
        if before_sp:
            before_sp = round(before_sp, 2)
        else:
            before_sp = 100.00

        # start pruning 
        print('start pruning model')
        pruning_model(model.module, args.percent, conv1=False)
        if_pruned = True

        current_mask = extract_mask(model.state_dict())
        remove_prune(model.module, conv1=False)

        # weight rewind
        model.module.load_state_dict(ticket_init_weight)

        prune_model_custom(model.module, current_mask, conv1=False)
        state_nat1, state_nat5 = validate(val_loader, model, criterion, args, writer, args.epochs)
        if args.attack_type is not 'None':
            state_val1, state_val5 = validate_adv(val_loader, model, criterion, args, writer, args.epochs)
        else:
            state_val1, state_val5 = -1, -1

        cur_sparsity = check_sparsity(model.module, use_mask=True, conv1=False)
        if cur_sparsity:
            cur_sparsity = round(cur_sparsity, 2)
        else:
            cur_sparsity = 100.00

        log.info("[PRUNE] Pruning Done! Sparsity %.2f --> %.2f", before_sp, cur_sparsity)
        log.info("[EVAL] Ticket eval: Nat@1: %.2f, Nat@5: %.2f", state_nat1, state_nat5)
        if args.attack_type is not 'None':
            log.info("[EVAL] Ticket eval: Adv@1: %.2f, Adv@5: %.2f", state_val1, state_val5)

        path = save_checkpoint({
            'mask': current_mask,
        }, False, checkpoint=args.ckpt_base_dir, filename=f"mask_state{prun_iter}_sp{cur_sparsity}.pth.tar")
        log.info(f"[CKPT] Mask saved at directory: {path}")

        print("*" * 150)
        log.info("*" * 150)
        print(f"State {prun_iter} report: ")
        log.info(f"State {prun_iter} report: ")
        print(f'Training: best acc1: {best_acc1}, NAABR: {natural_acc1_at_best_robustness}, best epoch: {best_epoch}')
        log.info(
            f'Training: best acc1: {best_acc1}, NAABR: {natural_acc1_at_best_robustness}, best epoch: {best_epoch}')
        print(f"Pruning: Sparsity before: {before_sp}, Sparsity after: {cur_sparsity}")
        log.info(f"Pruning: Sparsity before: {before_sp}, Sparsity after: {cur_sparsity}")
        print(f"Evaluation: Adv@1: {state_val1}, Adv@5: {state_val5}")
        log.info(f"Evaluation: Adv@1: {state_val1}, Adv@5: {state_val5}")

        args.start_epoch = 0
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)


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
        return trainer.train, trainer.validate, None, trainer.modifier
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


if __name__ == '__main__':
    main()
