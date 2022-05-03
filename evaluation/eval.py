import abc
import argparse
import logging
import os
import pathlib
import random
import shutil
import time
import warnings
import _thread
import copy
import math

import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from ignite.engine import Engine
from robustness.tools.helpers import has_attr
import tqdm
from ignite.contrib.metrics import ROC_AUC
from sklearn.metrics import roc_auc_score

import models
from matplotlib import pyplot as plt
from robustness.tools import helpers
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
parser.add_argument('--set', type=str, default='cifar10',
                    help='ImageNet, cifar10, cifar100, svhn, caltech101, dtd, flowers, pets, sun')
parser.add_argument('--prune_method', type=str, default='imp', help='omp, omp_s, imp, rst')
parser.add_argument('--prune_rate', type=float, default=0.2, help='prune ratio')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--log_dir', default='runs', type=str)
parser.add_argument('--name', default='debug_runs', type=str, help='experiment name')
parser.add_argument('--weight_dir', type=str,
                    default='/home/yuanye/RST/imp/runs/R18_c10_Linf_Eps2_sp80.0_c10/checkpoints/model_best.pth.tar',
                    help='path of the pretrained weight')
parser.add_argument('--mask_dir', type=str,
                    default=None,
                    help='path of the extracted mask')
parser.add_argument('--pytorch-pretrained', action='store_true',
                    help='If True, loads a Pytorch pretrained natural weight.')
parser.add_argument('--random', action="store_true", help="using random-init model")
parser.add_argument('--conv1', action="store_true", help="")
parser.add_argument('--attack_type', default='None', choices=['fgsm', 'fgsm-rs', 'pgd', 'free', 'None'])
parser.add_argument('--constraint', default="Linf", type=str)
parser.add_argument('--epsilon', default=3, type=float)
parser.add_argument('--amplitude', default=50, type=float, help="amplitude of gaussaine noise")

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

############################# scp settings ################################
parser.add_argument('--use_scp', action='store_true', help="scp the ckpts to target host", default=True)
parser.add_argument('--remove_local_ckpt', action='store_true',
                    help="remove the local checkpoints for disk space saving",
                    default=False)
parser.add_argument('--remote_dir', type=str, default='sw99@eic-2020gpu6.ece.rice.edu:/data1/sw99/remote_ckpt/imp',
                    help='path to scp the model, make sure you have all keys set')


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
    model, val_loader, val_loader_OoD = get_model_dataset(args)
    if_pruned = False

    if args.per_class_accuracy:
        assert args.set in ['pets', 'caltech101', 'flowers'], \
            f'Per-class accuracy not supported for the {args.set} dataset.'

        # VERY IMPORTANT
        # We report the per-class accuracy using the validation
        # set distribution. So ignore the training accuracy (as you will see it go
        # beyond 100. Don't freak out, it doesn't really capture anything),
        # just look at the validation accuarcy
        log.info("[INIT] => using per_class_accuracy")
        log.info(
            """
                          # VERY IMPORTANT
        # We report the per-class accuracy using the validation
        # set distribution. So ignore the training accuracy (as you will see it go
        # beyond 100. Don't freak out, it doesn't really capture anything),
        # just look at the validation accuracy
            """
        )
        args.custom_accuracy = get_per_class_accuracy(args, val_loader)

    writer = SummaryWriter(log_dir=log_base_dir)

    # Load pretrained weights
    if args.weight_dir:
        log.info("Loading pretrained weights")
        print("=> loading checkpoint '{}'".format(args.weight_dir))
        log.info("[LOAD] => loading checkpoint '{}'".format(args.weight_dir))
        checkpoint = torch.load(args.weight_dir)

        new_state_dict = OrderedDict()
        new_state_dict_no_fc = OrderedDict()
        sd = checkpoint['state_dict']
        for k, v in sd.items():
            #name = k[len('module.'):]
            if 'attacker' in k:
                break
            if 'normalize' not in k:
                new_state_dict[k] = v
            if 'normalize' not in k and 'fc' not in k:
                new_state_dict_no_fc[k] = v

        model.load_state_dict(new_state_dict, strict=False)

        # except:
        #     model.load_state_dict(new_state_dict_no_fc, strict=True)

    else:
        log.info("[LOAD] => Pytorch natural pretrained model")

    print('dataparallel mode')
    log.info("[INIT] dataparallel mode")
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.prune_method == "omp":
        print('execute OMP prune rate: {}'.format(args.prune_rate))
        pruning_model(model.module, args.prune_rate, conv1=args.conv1)
        remove_prune(model.module, conv1=args.conv1)
        check_sparsity(model.module, use_mask=False, conv1=args.conv1)

    elif args.mask_dir is not None:
        # Load mask
        log.info("[LOAD] => loading mask '{}'".format(args.mask_dir))
        mask = torch.load(args.mask_dir)
        args.mask = mask['mask']
        prune_model_custom(model.module, args.mask, False)
        remove_prune(model.module, False)
        check_sparsity(model.module, use_mask=False, conv1=args.conv1)

    # Linear evaluation
    if args.linear_eval:
        log.info("[INIT] => Linear Evaluation Mode")
        freeze_model(log, model.module, 4)

    cudnn.benchmark = True

    # I. Adversarial Robustness
    adv_acc1, adv_acc5 = validate_adv(val_loader, model.module, criterion, args, writer)
    log.info("[EVAL] Adversarial Robustness: Adv@1: %.2f, Adv@5: %.2f", adv_acc1, adv_acc5)

    # II. OoD Detection
    ovr, ovo = validate(val_loader_OoD, model.module, criterion, args, writer)
    log.info("[EVAL] OoD Detection: OVR: %.2f, OVO: %.2f", ovr, ovo)
    """
    References:
    OVR: Stands for One-vs-rest. Computes the AUC of each class against the rest. This treats the multiclass case in the same way as the multilabel case.
    OVO: Stands for One-vs-one. Computes the average AUC of all possible pairwise combinations of classes
    """

    # III. Natural Perturbation





    # IV. Uncertainty




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
        args.per_class_accuracy = False
        train_loader, data_norm, test_loader = cifar10_dataloaders(args, use_val=False, norm=False)
    elif args.set == 'cifar100':
        args.classes = 100
        args.per_class_accuracy = False
        train_loader, data_norm, test_loader = cifar100_dataloaders(args, use_val=False, norm=False)
    elif args.set == 'svhn':
        args.classes = 10
        args.per_class_accuracy = False
        train_loader, data_norm, test_loader = svhn_dataloaders(args, use_val=False, norm=False)
    elif args.set == 'ImageNet':
        args.classes = 1000
        args.per_class_accuracy = False
        train_loader, data_norm, test_loader = imagenet_dataloaders(args, use_val=False, norm=False)
    elif args.set == 'caltech101':
        args.classes = 102
        args.per_class_accuracy = True
        train_loader, data_norm, test_loader = caltech101_dataloaders(args, use_val=False, norm=False)
    elif args.set == 'dtd':
        args.classes = 47
        args.per_class_accuracy = False
        train_loader, data_norm, test_loader = dtd_dataloaders(args, use_val=False, norm=False)
    elif args.set == 'flowers':
        args.classes = 102
        args.per_class_accuracy = True
        train_loader, data_norm, test_loader = flowers_dataloaders(args, use_val=False, norm=False)
    elif args.set == 'pets':
        args.classes = 37
        args.per_class_accuracy = True
        train_loader, data_norm, test_loader = pets_dataloaders(args, use_val=False, norm=False)
    elif args.set == 'sun':
        args.classes = 397
        args.per_class_accuracy = False
        train_loader, data_norm, test_loader = SUN397_dataloaders(args, use_val=False, norm=False)
    else:
        raise ValueError("Unknown Dataset")

    # prepare model
    # model = models.__dict__[args.arch](pretrained=(not args.random), normalize=data_norm)
    if args.arch == 'resnet18':
        model = models.resnet.resnet18(pretrained=False, normalize=data_norm, num_classes=args.classes)
    elif args.arch == 'resnet50':
        model = models.resnet.resnet50(pretrained=False, normalize=data_norm, num_classes=args.classes)
    else:
        print('Wrong Model Arch')
        exit()
    return model, train_loader, test_loader


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
        return torch.Tensor(class_weights)

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

        return weighted_prec1, normal_prec1

    return custom_acc


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


def validate_adv(val_loader, model, criterion, args, writer):
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2471, 0.2435, 0.2616)

    svhn_mean = (0.5, 0.5, 0.5)
    svhn_std = (0.5, 0.5, 0.5)

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    if args.set == 'ImageNet':
        mean = imagenet_mean
        std = imagenet_std
    elif args.set == 'svhn':
        mean = svhn_mean
        std = svhn_std
    elif 'cifar' in args.set:
        mean = cifar_mean
        std = cifar_std
    else:
        print('Plz specify mean and std in the trainer for dataset:', args.set)
        exit()

    mu = torch.tensor(mean).view(3, 1, 1).cuda()
    std = torch.tensor(std).view(3, 1, 1).cuda()

    upper_limit = ((1 - mu) / std)
    lower_limit = ((0 - mu) / std)

    # epsilon = (args.epsilon / 255.) / std
    # # alpha = (args.alpha / 255.) / std
    # alpha = (2 / 255.) / std

    if args.constraint == 'Linf':
        epsilon = args.epsilon / 255.
        alpha = 2 / 255.
    elif args.constraint == 'L2':
        epsilon = args.epsilon
        alpha = 2 / 255.
    else:
        epsilon = args.epsilon / 255.
        alpha = epsilon * 2.5 / 5

    end = time.time()
    for i, (X, y) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
    ):

        X = X.cuda()
        y = y.cuda()

        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, lower_limit, upper_limit, args, attack_iters=3, restarts=1)
        # compute output
        output = model(X + pgd_delta)

        loss = criterion(output, y)

        # measure accuracy and record loss
        model_logits = output[0] if (type(output) is tuple) else output
        if has_attr(args, "custom_accuracy"):
            # print("using custom accuracy")
            acc1, acc5 = args.custom_accuracy(model_logits, y)
        else:
            acc1, acc5 = accuracy(model_logits, y, topk=(1, 5))

        losses.update(loss.item(), X.size(0))
        top1.update(acc1.item(), X.size(0))
        top5.update(acc5.item(), X.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    progress.display(len(val_loader))

    return top1.avg, top5.avg


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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
        return torch.Tensor(class_weights)

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

        return weighted_prec1, normal_prec1

    return custom_acc


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_pgd(model, X, y, epsilon, alpha, lower_limit, upper_limit, args, attack_iters=20, restarts=1):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    ones = torch.ones((3, 1, 1)).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(3):
            delta[:, i, :, :].uniform_(-epsilon, epsilon)
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            loss = F.cross_entropy(output, y)
            loss.backward()

            grad = delta.grad.detach()
            if args.constraint == 'Linf':
                delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon * ones, epsilon * ones)
                delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            elif args.constraint == 'L2':
                l = len(X.shape) - 1
                g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, *([1] * l))
                scaled_g = grad / (g_norm + 1e-10)
                delta.data = delta.data + alpha * scaled_g
                delta.data = delta.renorm(p=2, dim=0, maxnorm=epsilon)
                delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            else:
                print('Wrong args.constraint')
                exit()

        with torch.no_grad():
            all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)

    return max_delta


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, tqdm_writer=True):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if not tqdm_writer:
            print("\t".join(entries))
        else:
            tqdm.tqdm.write("\t".join(entries))

    def write_to_tensorboard(
            self, writer: SummaryWriter, prefix="train", global_step=None
    ):
        for meter in self.meters:
            avg = meter.avg
            val = meter.val
            if meter.write_val:
                writer.add_scalar(
                    f"{prefix}/{meter.name}_val", val, global_step=global_step
                )

            if meter.write_avg:
                writer.add_scalar(
                    f"{prefix}/{meter.name}_avg", avg, global_step=global_step
                )

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class Meter(object):
    @abc.abstractmethod
    def __init__(self, name, fmt=":f"):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def update(self, val, n=1):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass


class AverageMeter(Meter):
    """ Computes and stores the average and current value """

    def __init__(self, name, fmt=":f", write_val=True, write_avg=True):
        self.name = name
        self.fmt = fmt
        self.reset()

        self.write_val = write_val
        self.write_avg = write_avg

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class VarianceMeter(Meter):
    def __init__(self, name, fmt=":f", write_val=False):
        self.name = name
        self._ex_sq = AverageMeter(name="_subvariance_1", fmt=":.02f")
        self._sq_ex = AverageMeter(name="_subvariance_2", fmt=":.02f")
        self.fmt = fmt
        self.reset()
        self.write_val = False
        self.write_avg = True

    @property
    def val(self):
        return self._ex_sq.val - self._sq_ex.val ** 2

    @property
    def avg(self):
        return self._ex_sq.avg - self._sq_ex.avg ** 2

    def reset(self):
        self._ex_sq.reset()
        self._sq_ex.reset()

    def update(self, val, n=1):
        self._ex_sq.update(val ** 2, n=n)
        self._sq_ex.update(val, n=n)

    def __str__(self):
        return ("{name} (var {avg" + self.fmt + "})").format(
            name=self.name, avg=self.avg
        )


def validate(val_loader, model, criterion, args, writer):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    ovr = AverageMeter("Acc@1", ":6.2f", write_val=False)
    ovo = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, ovr, ovo], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (X, y) in tqdm.tqdm(
                enumerate(val_loader), ascii=True, total=len(val_loader)
        ):

            X = X.cuda()

            y = y.cuda()

            # compute output
            output = model(X)

            loss = criterion(output, y)
            # probs = F.softmax(output).detach().cpu().numpy()[0]
            probs = F.softmax(output, dim=1).detach().cpu()

            # measure accuracy and record loss
            # model_logits = output[0] if (type(output) is tuple) else output


            # if has_attr(args, "custom_accuracy"):
            #     # print("using custom accuracy")
            #     acc1, acc5 = args.custom_accuracy(model_logits, y)
            # else:
            #     acc1, acc5 = accuracy(model_logits, y, topk=(1, 5))

            losses.update(loss.item(), X.size(0))
            try:
                cur_ovr = roc_auc_score(y_true=y.cpu(), y_score=probs, multi_class='ovr')
                ovr.update(cur_ovr, X.size(0))
            except ValueError:
                pass
            try:
                cur_ovo = roc_auc_score(y_true=y.cpu(), y_score=probs, multi_class='ovo')
                ovo.update(cur_ovo, X.size(0))
            except ValueError:
                pass

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))

    return ovr.avg, ovo.avg


if __name__ == '__main__':
    main()
