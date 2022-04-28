import pathlib
from collections import OrderedDict

from torch.nn.utils import prune

import network
import utils
import os
import random
import args
import numpy as np
import time
from torch.utils import data
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import pdb
import pruning


def main():
    opts = args.get_argparser().parse_args()
    args.print_args(opts)

    os.makedirs(opts.log_dir, exist_ok=True)
    opts.run_base_dir, opts.ckpt_base_dir, opts.log_base_dir = get_directories(opts)
    os.makedirs(opts.ckpt_base_dir, exist_ok=True)

    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst = args.get_dataset(opts)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride,
                                  pretrained_backbone=opts.pytorch_pretrained)

    if opts.model_path:
        print("=> loading checkpoint '{}'".format(opts.model_path))
        checkpoint = torch.load(opts.model_path)

        new_state_dict = OrderedDict()
        sd = checkpoint['model']
        for k, v in sd.items():
            if 'attacker' in k:
                break
            if 'normalize' not in k:
                name = k[len('module.model.'):]
                new_state_dict[name] = v
        model.backbone.load_state_dict(new_state_dict, strict=False)
        print("=> loaded checkpoint '{}' (epoch {})".format(opts.model_path, checkpoint['epoch']))
    elif opts.pytorch_pretrained:
        print("[LOAD] => Pytorch natural pretrained model")
    else:
        print("Random initialized ResNet50 as classification backbone")

    # Prune the backbone model
    if not opts.full_model_transfer:
        if opts.prune_method.lower() == "omp":
            print('execute OMP prune rate: {}'.format(opts.prune_rate))
            pruning_model(model.backbone, opts.prune_rate, conv1=opts.conv1)
            check_sparsity(model.backbone, use_mask=True, conv1=opts.conv1)

        else:
            print(f"Prune Method: {opts.prune_method} | Mask DIR:[{opts.mask_dir}]")
            mask = torch.load(opts.mask_dir, map_location="cuda")
            opts.mask = mask['mask']
            prune_model_custom(model.backbone, opts.mask, conv1=opts.conv1)
            check_sparsity(model.backbone, use_mask=True, conv1=opts.conv1)

    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)
    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    writer = SummaryWriter(log_dir=str(opts.log_base_dir))
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cuda'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = args.validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    # print("Transfer Mask dir:[{}]".format(opts.mask_dir))
    # mask_dict = torch.load(opts.mask_dir, map_location='cuda')
    # pruning.imagenet_pruning_model_custom_res50v1(model.module.backbone, mask_dict)
    # pruning.see_zero_rate(model.module.backbone)

    interval_loss = 0
    total_time = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:

            t0 = time.time()
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
                      "Epoch:[{}], Itrs:[{}/{}], Loss:[{:.4f}], Time:[{:.4f} min]"
                      .format(cur_epochs, cur_itrs, int(opts.total_itrs), interval_loss, total_time / 60))
                writer.add_scalar('Loss/train', interval_loss, cur_itrs)
                interval_loss = 0.0
                total_time = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                # save_ckpt(os.path.join(opts.ckpt_base_dir, 'latest_%s_%s_os%d.pth' %
                #                        (opts.model, opts.dataset, opts.output_stride)))
                print("validation...")
                model.eval()
                val_score, ret_samples = args.validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))

                writer.add_scalar('mIOU/test', val_score['Mean IoU'], cur_itrs)

                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    # save_ckpt(os.path.join(opts.ckpt_base_dir, 'best_%s_%s_os%d.pth' %
                    #                        (opts.model, opts.dataset, opts.output_stride)))

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()

            scheduler.step()
            t1 = time.time()
            total_time += t1 - t0

            if cur_itrs >= opts.total_itrs:
                writer.close()
                print("syd ----------[imagenet]----------------")
                print(f"eps{opts.eps}")
                print("syd Last IOU:[{:.6f}]".format(val_score['Mean IoU']))
                print("syd Best IOU:[{:.6f}]".format(best_score))
                print("syd --------------------------")

                sp = str(opts.prune_rate) if opts.prune_method == "omp" else \
                    opts.mask_dir.split('/')[-1].split('_')[-1][:-len('.pth.tar')]
                outp_str = (
                               "nat" if opts.pytorch_pretrained else f"adv{opts.eps}") + opts.name + " " + opts.prune_method + " " + sp + f" Last IOU={val_score['Mean IoU']:.6f} Best IOU={best_score:.6f} \n"
                print(outp_str)
                file_name = f"result_log.txt"
                f = open(file_name, "a+")
                f.write(outp_str)
                f.close()

                return


def get_directories(args):
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{args.prune_method}/{args.name}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{args.prune_method}/{args.name}"
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


def check_sparsity(model, use_mask=True, conv1=True):
    sum_list = 0
    zero_sum = 0

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
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
                zero_sum += torch.sum(buffer == 0).item()

    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name:
                sum_list += param.nelement()
                zero_sum += torch.sum(param == 0).item()

    return sum_list, zero_sum


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


if __name__ == '__main__':
    main()
