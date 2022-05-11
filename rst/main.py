import os
import pathlib
import random
import time
import shutil
import math
from collections import OrderedDict

import numpy as np
from robustness.tools import helpers
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tqdm import tqdm

from utils import builder
from utils.conv_type import FixedSubnetConv, SampleSubnetConv, SubnetConv
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import (
    set_model_prune_rate,
    freeze_model_weights,
    freeze_model_subnet,
    save_checkpoint,
    get_lr,
    LabelSmoothing,
    init_model_weight_with_score,
)
from utils.schedulers import get_policy
import logging

from args import args
import importlib

from utils.builder import get_builder
from data.dataset import *
from models import *


def main():
    # print(args)
    if args.prune_percent:
        args.prune_rate = args.prune_percent / 100
        print("current prune_rate=", args.prune_rate)

    if args.pytorch_pretrained:
        args.pretrained = None

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    # Set up directories
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    args.ckpt_base_dir = ckpt_base_dir

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

    if args.attack_type == 'free' and args.set == 'ImageNet':
        args.lr_policy = 'multistep_lr_imagenet_free'
        args.epochs = int(math.ceil(args.epochs / args.n_repeats))

    train, validate, validate_adv, modifier = get_trainer(args)
    model, train_loader, val_loader = get_model_dataset(args)
    # data = get_dataset(args)
    #
    # # create model and optimizer
    # model = get_model(args, data.data_norm)

    if args.per_class_accuracy:
        assert args.set in ['pets', 'caltech101', 'flowers', 'cars'], \
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

    if args.task != 'search':
        if args.pretrained is None:
            path = run_base_dir.parent / 'search' / 'checkpoints' / 'model_best.pth'
            if os.path.exists(path):
                args.pretrained = path
            else:
                path = run_base_dir.parent / 'checkpoints' / 'model_best.pth'
                if os.path.exists(path):
                    args.pretrained = path
                else:
                    print('No pretrained checkpoint:', path)
                    exit()
        pretrained(args, model)

    elif args.pretrained:
        pretrained(args, model)

    # init mask
    for n, m in model.named_modules():
        if isinstance(m, SubnetConv):
            m.init_score_with_weight_mag_with_scale()
            args.log.info(f"Init model {n} mask with weight. ")

    if args.classes != 1000:
        change_fc_layer(args, model)

    # freezing the weights if we are only doing subnet training
    if args.task == 'search':
        freeze_model_weights(model)

    else:
        # freezing the subnet and finetuning the model weights
        freeze_model_subnet(model)

        # finetune the robust ticket with random weight intialization
        if args.task == 'ft_reinit':
            args.init = args.ft_init
            builder = get_builder()
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    builder._init_conv(module)

        # finetune the whole model with the initialization to be the robust ticket
        if args.task == 'ft_full':
            init_model_weight_with_score(model, prune_rate=1-args.prune_rate)
            # set_model_prune_rate(model, prune_rate=1.0)
            if args.freeze_level != -1:
                freeze_model(log, model, freeze_level=args.freeze_level)
   
    model = set_gpu(args, model)
    optimizer = get_optimizer(args, model)
    lr_policy = get_policy(args.lr_policy)(optimizer, args)

    if args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = LabelSmoothing(smoothing=args.label_smoothing)

    # optionally resume from a checkpoint
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    # natural_acc1_at_best_robustness = None

    if args.automatic_resume:
        args.resume = ckpt_base_dir / 'model_latest.pth'
        if os.path.isfile(args.resume):
            best_acc1 = resume(args, model, optimizer)
            log.info("Resumed from ckpt. ")
        else:
            log.info('Train from scratch.')

    elif args.resume:
        best_acc1 = resume(args, model, optimizer)

    # Data loading code
    if args.evaluate:
        if args.attack_type != 'None':
            acc1, acc5 = validate_adv(
                val_loader, model, criterion, args, writer=None, epoch=args.start_epoch
            )

            natural_acc1, natural_acc5 = validate(
                val_loader, model, criterion, args, writer=None, epoch=args.start_epoch
            )

            log.info('Natural Acc: %.2f, Robust Acc: %.2f', natural_acc1, acc1)

        else:
            acc1, acc5 = validate(
                val_loader, model, criterion, args, writer=None, epoch=args.start_epoch
            )

            log.info('Natural Acc: %.2f', acc1)

        return

    writer = SummaryWriter(log_dir=log_base_dir)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time], prefix="Overall Timing"
    )

    end_epoch = time.time()
    args.start_epoch = args.start_epoch or 0
    acc1 = None

    # Save the initial state
    save_checkpoint(
        {
            "epoch": 0,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_acc1": best_acc1,
            "best_acc5": best_acc5,
            "best_train_acc1": best_train_acc1,
            "best_train_acc5": best_train_acc5,
            # 'natural_acc1_at_best_robustness': natural_acc1_at_best_robustness,
            "optimizer": optimizer.state_dict(),
            "curr_acc1": acc1 if acc1 else "Not evaluated",
        },
        False,
        filename=ckpt_base_dir / f"initial.state",
        save=False,
    )

    if args.discard_mode and args.progressive_prune:
        set_model_prune_rate(model, prune_rate=0.9)

    # Start training
    for epoch in range(args.start_epoch, args.epochs):
        lr_policy(epoch, iteration=None)
        modifier(args, epoch, model)

        cur_lr = get_lr(optimizer)
        print(cur_lr)

        # train for one epoch
        start_train = time.time()
        train_acc1, train_acc5 = train(
            train_loader, model, criterion, optimizer, epoch, args, writer=writer, log=log
        )

        if args.discard_mode:
            if (epoch + 1) % args.discard_epoch == 0:
                for n, m in model.named_modules():
                    if hasattr(m, "discard_low_score"):
                        m.discard_low_score(min(args.discard_rate * ((epoch + 1) // args.discard_epoch), 1))

        train_time.update((time.time() - start_train) / 60)

        # if 'ImageNet' in args.set:
        #     start_epoch = 30
        # else:
        #     start_epoch = 60

        # if args.optimizer == 'sgd':
        #     val_every = args.val_every if epoch > start_epoch else 10
        # else:
        #     val_every = args.val_every

        if epoch % args.val_every == 0 or epoch == args.epochs - 1:
            # evaluate on validation set
            start_validation = time.time()

            # if args.attack_type != 'None':
            #     acc1, acc5 = validate_adv(data.val_loader, model, criterion, args, writer, epoch)
            #     natural_acc1, natural_acc5 = validate(data.val_loader, model, criterion, args, writer, epoch)
            # else:
            #     acc1, acc5 = validate(data.val_loader, model, criterion, args, writer, epoch)

            acc1, acc5 = validate(val_loader, model, criterion, args, writer, epoch)

            validation_time.update((time.time() - start_validation) / 60)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            best_acc5 = max(acc5, best_acc5)
            best_train_acc1 = max(train_acc1, best_train_acc1)
            best_train_acc5 = max(train_acc5, best_train_acc5)

            # if is_best and args.attack_type != 'None':
            #     natural_acc1_at_best_robustness = natural_acc1

            if is_best:
                log.info(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")

            if is_best or epoch == args.epochs - 1:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": args.arch,
                        "state_dict": model.state_dict(),
                        "best_acc1": best_acc1,
                        "best_acc5": best_acc5,
                        "best_train_acc1": best_train_acc1,
                        "best_train_acc5": best_train_acc5,
                        # "natural_acc1_at_best_robustness": natural_acc1_at_best_robustness,
                        "optimizer": optimizer.state_dict(),
                        "curr_acc1": acc1,
                        "curr_acc5": acc5,
                    },
                    is_best,
                    filename=ckpt_base_dir / f"epoch_{epoch}.state",
                    save=False,
                )

            # if args.attack_type != 'None':
            #     log.info(
            #         'Epoch[%d][%d] curr natural acc: %.2f, natural acc at best robustness: %.2f \n curr robust acc: %.2f, best robust acc: %.2f',
            #         args.epochs, epoch, natural_acc1, natural_acc1_at_best_robustness, acc1, best_acc1)
            # else:
            #     log.info('Epoch[%d][%d] curr acc: %.2f, best acc: %.2f', args.epochs, epoch, acc1, best_acc1)

        log.info('Epoch[%d][%d] curr acc: %.2f, best acc: %.2f', args.epochs, epoch, acc1, best_acc1)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "best_acc5": best_acc5,
                "best_train_acc1": best_train_acc1,
                "best_train_acc5": best_train_acc5,
                # "natural_acc1_at_best_robustness": natural_acc1_at_best_robustness,
                "optimizer": optimizer.state_dict(),
                "curr_acc1": None,
                "curr_acc5": None,
            },
            is_best=False,
            filename=ckpt_base_dir / f"checkpoint.state",
            save=False,
        )

        # if args.conv_type == "SampleSubnetConv":
        #     count = 0
        #     sum_pr = 0.0
        #     for n, m in model.named_modules():
        #         if isinstance(m, SampleSubnetConv):
        #             # avg pr across 10 samples
        #             pr = 0.0
        #             for _ in range(10):
        #                 pr += (
        #                     (torch.rand_like(m.clamped_scores) >= m.clamped_scores)
        #                     .float()
        #                     .mean()
        #                     .item()
        #                 )
        #             pr /= 10.0
        #             writer.add_scalar("pr/{}".format(n), pr, epoch)
        #             sum_pr += pr
        #             count += 1

        #     args.prune_rate = sum_pr / count
        #     writer.add_scalar("pr/average", args.prune_rate, epoch)

        if args.discard_mode:
            if (epoch + 1) % args.discard_epoch == 0:
                if args.progressive_prune:
                    set_model_prune_rate(model,
                                         prune_rate=0.9 - min(args.discard_rate * ((epoch + 1) // args.discard_epoch),
                                                              1))

        epoch_time.update((time.time() - end_epoch) / 60)
        progress_overall.display(epoch)
        progress_overall.write_to_tensorboard(
            writer, prefix="diagnostics", global_step=epoch
        )

        writer.add_scalar("test/lr", cur_lr, epoch)
        end_epoch = time.time()

    # write_result_to_csv(
    #     best_acc1=best_acc1,
    #     best_acc5=best_acc5,
    #     best_train_acc1=best_train_acc1,
    #     best_train_acc5=best_train_acc5,
    #     prune_rate=args.prune_rate,
    #     curr_acc1=acc1,
    #     curr_acc5=acc5,
    #     base_config=args.config,
    #     name=args.name,
    # )
    
    log.info(args)
    log_dir_new = 'logs/log_' + args.name
    if not os.path.exists(log_dir_new):
        os.makedirs(log_dir_new)

    shutil.copyfile(log_path, os.path.join(log_dir_new, 'log_' + args.task + '.txt'))


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


def set_gpu(args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"

    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    return model


def resume(args, model, optimizer):
    if os.path.isfile(args.resume):
        args.log.info(f"=> Loading checkpoint '{args.resume}'")

        checkpoint = torch.load(args.resume)
        if args.start_epoch is None:
            args.log.info(f"=> Setting new start epoch at {checkpoint['epoch']}")
            args.start_epoch = checkpoint["epoch"]

        best_acc1 = checkpoint["best_acc1"]
        # natural_acc1_at_best_robustness = checkpoint["natural_acc1_at_best_robustness"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        args.log.info(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        return best_acc1
        # return best_acc1, natural_acc1_at_best_robustness
    else:
        args.log.info(f"=> No checkpoint found at '{args.resume}'")
        exit()


def pretrained(args, model):
    if os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        pretrained = torch.load(args.pretrained)
        try:
            sd = pretrained["state_dict"]
        except:
            sd = pretrained['model']

        new_state_dict = OrderedDict()
        for k, v in sd.items():
            if 'attacker' in k:
                break
            if 'normalize' not in k:
                name = k[len('module.model.'):]
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)

        # if args.not_strict:
        #     model_state_dict = model.state_dict()
        #     for k, v in pretrained.items():
        #         if k not in model_state_dict or v.size() != model_state_dict[k].size():
        #             print("IGNORE:", k)
        #     pretrained = {
        #         k: v
        #         for k, v in pretrained.items()
        #         if (k in model_state_dict and v.size() == model_state_dict[k].size())
        #     }
        #     model_state_dict.update(pretrained)
        #     model.load_state_dict(model_state_dict, strict=False)
        #
        # else:
        #     model.load_state_dict(pretrained)

    else:
        print("=> no pretrained weights found at '{}'".format(args.pretrained))
        exit()




# def get_dataset(args):
#     print(f"=> Getting {args.set} dataset")
#     dataset = getattr(data, args.set)(args)
#
#     return dataset
#
#
# def get_model(args, data_norm):
#     if args.first_layer_dense:
#         args.first_layer_type = "DenseConv"
#
#     print("=> Creating model '{}'".format(args.arch))
#
#     if args.set == 'ImageNet' or args.set == 'TinyImageNet':
#         args.classes = 1000
#     elif args.set == 'CIFAR100':
#         args.classes = 100
#     elif args.set == 'CIFAR10':
#         args.classes = 10
#
#     if args.set == 'ImageNet':
#         print(data_norm)
#         model = models.__dict__[args.arch](num_classes=1000, norm=data_norm)
#     else:
#         model = models.__dict__[args.arch](num_classes=1000)
#
#     # applying sparsity to the network
#     if (
#             args.conv_type != "DenseConv"
#             and args.conv_type != "SampleSubnetConv"
#             and args.conv_type != "ContinuousSparseConv"
#     ):
#         if args.prune_rate < 0:
#             raise ValueError("Need to set a positive prune rate")
#
#         set_model_prune_rate(model, prune_rate=args.prune_rate)
#         print(
#             f"=> Rough estimate model params {sum(int(p.numel() * (1 - args.prune_rate)) for n, p in model.named_parameters() if not n.endswith('scores'))}"
#         )
#
#     # freezing the weights if we are only doing subnet training
#     # if args.freeze_weights:
#     #     freeze_model_weights(model)
#
#     return model


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
        train_loader, data_norm, test_loader = caltech101_dataloaders(args, use_val=False, norm=True)
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
    elif args.set == 'cars':
        args.classes = 196
        args.per_class_accuracy = True
        train_loader, data_norm, test_loader = cars_dataloaders(args, use_val=False, norm=False)
    elif args.set == 'sun':
        args.classes = 397
        args.per_class_accuracy = False
        train_loader, data_norm, test_loader = SUN397_dataloaders(args, use_val=False, norm=False)
    else:
        raise ValueError("Unknown Dataset")

    # prepare model
    # model = models.__dict__[args.arch](num_classes=1000, norm=data_norm)
    if args.arch == 'ResNet18':
        model = ResNet18(pretrained=args.pytorch_pretrained, num_classes=1000, norm=data_norm)
    elif args.arch == 'ResNet50':
        model = ResNet50(pretrained=args.pytorch_pretrained, num_classes=1000, norm=data_norm)
    else:
        print('Wrong Model Arch')
        exit()

    # applying sparsity to the network
    if (
            args.conv_type != "DenseConv"
            and args.conv_type != "SampleSubnetConv"
            and args.conv_type != "ContinuousSparseConv"
    ):
        if args.prune_rate < 0:
            raise ValueError("Need to set a positive prune rate")

        set_model_prune_rate(model, prune_rate=1-args.prune_rate)
        print(
            f"=> Rough estimate model params {sum(int(p.numel() * (1 - args.prune_rate)) for n, p in model.named_parameters() if not n.endswith('scores'))}"
        )

    return model, train_loader, test_loader


def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        # if args.task == 'ft_full' and args.ft_full_mode == 'only_zero':
        #     parameters = list(model.named_parameters())
        #     bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]

        #     rest_params = []
        #     for n, v in parameters:
        #         if 'bn' not in n and 'weight' in n and v.requires_grad:
        #             mask = v.abs() == 0
        #             rest_params.append(v[mask])
        #             print(v[mask].size(), v.size())

        #     optimizer = torch.optim.SGD(
        #         [
        #             {
        #                 "params": bn_params,
        #                 "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
        #             },
        #             {"params": rest_params, "weight_decay": args.weight_decay},
        #         ],
        #         args.lr,
        #         momentum=args.momentum,
        #         weight_decay=args.weight_decay,
        #         nesterov=args.nesterov,
        #     )

        # elif args.task == 'ft_full' and args.ft_full_mode == 'decay_on_zero':
        #     parameters = list(model.named_parameters())
        #     bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]

        #     rest_params_zero = []
        #     rest_params_nonzero = []
        #     for n, v in parameters:
        #         if 'bn' not in n and 'weight' in n and v.requires_grad:
        #             mask = v.abs() == 0
        #             rest_params_zero.append(v[mask])  
        #             rest_params_nonzero.append(v[1-mask])   

        #     optimizer = torch.optim.SGD(
        #         [
        #             {
        #                 "params": bn_params,
        #                 "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
        #             },
        #             {"params": rest_params_nonzero, "weight_decay": args.weight_decay},
        #             {"params": rest_params_zero, "weight_decay": args.weight_decay_on_zero}
        #         ],
        #         args.lr,
        #         momentum=args.momentum,
        #         weight_decay=args.weight_decay,
        #         nesterov=args.nesterov,
        #     )

        # else:
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]

        if args.task == 'ft_full' and args.ft_full_mode == 'decay_on_zero':
            param_weight_decay = 0
        else:
            param_weight_decay = args.weight_decay

        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                },
                {"params": rest_params, "weight_decay": param_weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )

    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

    return optimizer


def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()


def get_directories(args):
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")

    if args.task == "search":
        last = args.task
    else:
        last = args.task + '_' + args.set

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{args.name}/prune_rate={args.prune_rate}/{last}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{config}/{args.name}/prune_rate={args.prune_rate}/{last}"
        )
    if args.width_mult != 1.0:
        run_base_dir = run_base_dir / "width_mult={}".format(str(args.width_mult))

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


def write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / "results.csv"

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "Base Config, "
            "Name, "
            "Prune Rate, "
            "Current Val Top 1, "
            "Current Val Top 5, "
            "Best Val Top 1, "
            "Best Val Top 5, "
            "Best Train Top 1, "
            "Best Train Top 5\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{base_config}, "
                "{name}, "
                "{prune_rate}, "
                "{curr_acc1:.02f}, "
                "{curr_acc5:.02f}, "
                "{best_acc1:.02f}, "
                "{best_acc5:.02f}, "
                "{best_train_acc1:.02f}, "
                "{best_train_acc5:.02f}\n"
            ).format(now=now, **kwargs)
        )


def change_fc_layer(args, model):
    print(f"Change the fc layer to fit in dataset: {args.set}")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.classes)


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


if __name__ == "__main__":
    main()
