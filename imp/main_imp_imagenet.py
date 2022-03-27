import argparse
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

from pruning_utils import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
############################# required settings ################################
parser.add_argument('--data', metavar='DIR', default='/data1/ImageNet/ILSVRC/Data/CLS-LOC',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--log_dir', default='runs', type=str)
parser.add_argument('--name', default='debug_runs', type=str, help='experiment name')
parser.add_argument('--model-path', type=str, default=None, help='path of the pretrained weight')
parser.add_argument('--pytorch-pretrained', action='store_true',
                    help='If True, loads a Pytorch pretrained model.')
parser.add_argument('--percent', default=0.2, type=float, help='pruning rate for each iteration')
parser.add_argument('--states', default=19, type=int, help='number of iterative pruning states')
parser.add_argument('--start_state', default=0, type=int, help='number of iterative pruning states')
parser.add_argument('--random', action="store_true", help="using random-init model")
parser.add_argument("--trainer", type=str, default="default", help="cs, ss, or standard training")
parser.add_argument('--attack_type', default='fgsm', choices=['fgsm', 'fgsm-rs', 'pgd', 'free'])

############################# other settings ################################
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
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

    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    args.ckpt_base_dir = ckpt_base_dir
    os.makedirs(args.ckpt_base_dir, exist_ok=True)

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model with official pretrained weight or random initialization
    print("=> using model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=(not args.random))
    if_pruned = False

    writer = SummaryWriter(log_dir=log_base_dir)

    # if model-path is not none, load the pretrained weight from this path
    if args.model_path:
        print("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)

        # Makes us able to load models saved with legacy versions
        state_dict_path = 'model'
        if not ('model' in checkpoint):
            state_dict_path = 'state_dict'

        sd = checkpoint[state_dict_path]
        sd = {k[len('module.'):]: v for k, v in sd.items()}
        model.load_state_dict(sd)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))

    # init pretrianed weight 
    ticket_init_weight = copy.deepcopy(model.state_dict())
    save_checkpoint({
        'state_dict': model.state_dict(),
    }, False, checkpoint=args.ckpt_base_dir, filename="weight_init.pth.tar")

    print('dataparallel mode')
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            args.start_state = checkpoint['state']
            best_acc1 = checkpoint['best_acc1']
            if_pruned = checkpoint['if_pruned']
            ticket_init_weight = checkpoint['init_weight']

            if if_pruned:
                prune_model_custom(model.module, checkpoint['mask'], False)

            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    train, validate, validate_adv, modifier = get_trainer(args)

    if args.evaluate:
        nat_acc1, nat_acc5 = validate(val_loader, model, criterion, args, writer)
        adv_acc1, adv_acc5 = validate_adv(val_loader, model, criterion, args, writer)
        print("Evaluation result: ")
        print(f"Natural Acc1: {nat_acc1}, Natural Acc5: {nat_acc5}, Robust Acc1: {adv_acc1}, Robust Acc5: {adv_acc5}")
        return

    for prun_iter in range(args.start_state, args.states):
        cur_sparsity = check_sparsity(model.module, use_mask=False, conv1=False)
        cur_sparsity = round(cur_sparsity, 2)

        # the best record of the current pruning state
        best_acc1 = 0.0
        best_acc5 = 0.0
        best_nat_acc1 = 0.0
        best_nat_acc5 = 0.0
        best_train_acc1 = 0.0
        best_train_acc5 = 0.0
        best_epoch = 0
        natural_acc1_at_best_robustness = 0.0

        for epoch in range(args.start_epoch, args.epochs):
            print(optimizer.state_dict()['param_groups'][0]['lr'])
            # train for one epoch
            train_acc1, train_acc5 = train(train_loader, model, criterion, optimizer, epoch, args, writer)

            # evaluate on validation set
            nat_acc1, nat_acc5 = validate(val_loader, model, criterion, args, writer)

            # evaluate on adversary validation set
            adv_acc1, adv_acc5 = validate_adv(val_loader, model, criterion, args, writer)

            # remember best acc@1 and save checkpoint
            is_best = adv_acc1 > best_acc1
            best_acc1 = max(adv_acc1, best_acc1)

            if is_best:
                best_epoch = epoch + 1
                natural_acc1_at_best_robustness = nat_acc1

            if if_pruned:
                mask_dict = extract_mask(model.state_dict())
            else:
                mask_dict = None

            save_checkpoint({
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
            }, is_best, checkpoint=args.ckpt_base_dir, best_name='sparsity' + str(cur_sparsity) + 'model_best.pth.tar')

        check_sparsity(model.module, use_mask=False, conv1=False)
        print(f"State {prun_iter} training report: ")
        print('best acc1 = ', best_acc1, ' best epoch = ', best_epoch)

        # start pruning 
        print('start pruning model')
        pruning_model(model.module, args.percent, conv1=False)
        if_pruned = True

        current_mask = extract_mask(model.state_dict())
        remove_prune(model.module, conv1=False)

        # weight rewind
        model.module.load_state_dict(ticket_init_weight)

        prune_model_custom(model.module, current_mask, conv1=False)
        validate_adv(val_loader, model, criterion, args, writer)
        args.start_epoch = 0
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)


# def train(train_loader, model, criterion, optimizer, epoch, args):
#     batch_time = AverageMeter('Time', ':6.3f')
#     data_time = AverageMeter('Data', ':6.3f')
#     losses = AverageMeter('Loss', ':.4e')
#     top1 = AverageMeter('Acc@1', ':6.2f')
#     top5 = AverageMeter('Acc@5', ':6.2f')
#     progress = ProgressMeter(
#         len(train_loader),
#         [batch_time, data_time, losses, top1, top5],
#         prefix="Epoch: [{}]".format(epoch))
#
#     # switch to train mode
#     model.train()
#
#     end = time.time()
#     for i, (images, target) in enumerate(train_loader):
#         # measure data loading time
#         data_time.update(time.time() - end)
#
#         if args.gpu is not None:
#             images = images.cuda(args.gpu, non_blocking=True)
#         target = target.cuda(args.gpu, non_blocking=True)
#
#         # compute output
#         output = model(images)
#         loss = criterion(output, target)
#
#         # measure accuracy and record loss
#         acc1, acc5 = accuracy(output, target, topk=(1, 5))
#         losses.update(loss.item(), images.size(0))
#         top1.update(acc1[0], images.size(0))
#         top5.update(acc5[0], images.size(0))
#
#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         if i % args.print_freq == 0:
#             progress.display(i)
#
#
# def validate(val_loader, model, criterion, args):
#     batch_time = AverageMeter('Time', ':6.3f')
#     losses = AverageMeter('Loss', ':.4e')
#     top1 = AverageMeter('Acc@1', ':6.2f')
#     top5 = AverageMeter('Acc@5', ':6.2f')
#     progress = ProgressMeter(
#         len(val_loader),
#         [batch_time, losses, top1, top5],
#         prefix='Test: ')
#
#     # switch to evaluate mode
#     model.eval()
#
#     with torch.no_grad():
#         end = time.time()
#         for i, (images, target) in enumerate(val_loader):
#             if args.gpu is not None:
#                 images = images.cuda(args.gpu, non_blocking=True)
#             target = target.cuda(args.gpu, non_blocking=True)
#
#             # compute output
#             output = model(images)
#             loss = criterion(output, target)
#
#             # measure accuracy and record loss
#             acc1, acc5 = accuracy(output, target, topk=(1, 5))
#             losses.update(loss.item(), images.size(0))
#             top1.update(acc1[0], images.size(0))
#             top5.update(acc5[0], images.size(0))
#
#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()
#
#             if i % args.print_freq == 0:
#                 progress.display(i)
#
#         # TODO: this should also be done with the ProgressMeter
#         print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
#               .format(top1=top1, top5=top5))
#
#     return top1.avg


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar', best_name='model_best.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, best_name))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir


if __name__ == '__main__':
    main()
