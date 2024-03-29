import logging
import pathlib

from torch import nn
from torch.nn.utils import prune

import utils.gpu as gpu
from model.build_model import Build_Model
from model.loss.yolo_loss import YoloV4Loss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import utils.datasets as data
import time
import random
import argparse
from eval.evaluator import *
from utils.tools import *
from tensorboardX import SummaryWriter
import config.yolov4_config as cfg
from utils import cosine_lr_scheduler
from utils.log import Logger
# from apex import amp
import torchvision.models as models
from eval_coco import *
from eval.cocoapi_evaluator import COCOAPIEvaluator
import pdb
import ap
import random
import pruning
import numpy as np


def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
    return torch.stack(imgs, 0), targets


def seed_setup(args):
    print('INFO: Seeds : [{}]'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


class Trainer(object):
    def __init__(self, weight_path, resume, gpu_id, accumulate, fp_16, args):
        init_seeds(0)
        seed_setup(args)
        self.args = args
        self.fp_16 = fp_16
        self.device = gpu.select_device(gpu_id)
        self.start_epoch = 0
        self.best_mAP = 0.0
        self.accumulate = accumulate
        self.weight_path = weight_path
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        if self.multi_scale_train:
            print("Using multi scales training")
        else:
            print("train img size is {}".format(cfg.TRAIN["TRAIN_IMG_SIZE"]))
        self.train_dataset = data.Build_Dataset(
            anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"]
        )
        self.epochs = (
            cfg.TRAIN["YOLO_EPOCHS"]
            if cfg.MODEL_TYPE["TYPE"] == "YOLOv4"
            else cfg.TRAIN["Mobilenet_YOLO_EPOCHS"]
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=cfg.TRAIN["BATCH_SIZE"],
            num_workers=cfg.TRAIN["NUMBER_WORKERS"],
            shuffle=True,
            pin_memory=True,
        )

        self.yolov4 = Build_Model(weight_path=weight_path, resume=resume, pretrained=self.args.pytorch_pretrained).to(
            self.device)
        self.optimizer = optim.SGD(
            self.yolov4.parameters(),
            lr=cfg.TRAIN["LR_INIT"],
            momentum=cfg.TRAIN["MOMENTUM"],
            weight_decay=cfg.TRAIN["WEIGHT_DECAY"],
        )

        self.criterion = YoloV4Loss(
            anchors=cfg.MODEL["ANCHORS"],
            strides=cfg.MODEL["STRIDES"],
            iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"],
        )

        self.scheduler = cosine_lr_scheduler.CosineDecayLR(
            self.optimizer,
            T_max=self.epochs * len(self.train_dataloader),
            lr_init=cfg.TRAIN["LR_INIT"],
            lr_min=cfg.TRAIN["LR_END"],
            warmup=cfg.TRAIN["WARMUP_EPOCHS"] * len(self.train_dataloader),
        )
        if resume:
            self.__load_resume_weights(weight_path)

    def __load_resume_weights(self, weight_path):

        last_weight = os.path.join(os.path.split(weight_path)[0], "last.pt")
        chkpt = torch.load(last_weight, map_location=self.device)
        self.yolov4.load_state_dict(chkpt["model"])

        self.start_epoch = chkpt["epoch"] + 1
        if chkpt["optimizer"] is not None:
            self.optimizer.load_state_dict(chkpt["optimizer"])
            self.best_mAP = chkpt["best_mAP"]
        del chkpt

    def __save_model_weights(self, epoch, mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
        best_weight = os.path.join(
            os.path.split(self.weight_path)[0], "best.pt"
        )
        last_weight = os.path.join(
            os.path.split(self.weight_path)[0], "last.pt"
        )
        chkpt = {
            "epoch": epoch,
            "best_mAP": self.best_mAP,
            "model": self.yolov4.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(chkpt, last_weight)

        if self.best_mAP == mAP:
            torch.save(chkpt["model"], best_weight)

        if epoch > 0 and epoch % 10 == 0:
            torch.save(
                chkpt,
                os.path.join(
                    os.path.split(self.weight_path)[0],
                    "backup_epoch%g.pt" % epoch,
                ),
            )
        del chkpt

    def train(self):
        global writer
        logger.info(
            "Training start,img size is: {:d},batchsize is: {:d},work number is {:d}".format(
                cfg.TRAIN["TRAIN_IMG_SIZE"],
                cfg.TRAIN["BATCH_SIZE"],
                cfg.TRAIN["NUMBER_WORKERS"],
            )
        )
        logger.info("Train datasets number is : {}".format(len(self.train_dataset)))

        print('Loading base network...')
        if self.args.transfer_type == 'imagenet':
            print("-" * 100)
            print("Use resnet50 [imagnet] weight")
            print("-" * 100)

        # elif self.args.transfer_type == 'moco':
        #     print("-" * 100)
        #     print("load resnet50 [moco] weight")
        #     print("-" * 100)
        #     moco_ckpt = torch.load('./weight/moco_v2_800ep_pretrain.pth.tar', map_location='cuda')['state_dict']
        #     resnet50_pytorch = models.resnet50(pretrained=False)
        #     moco_state_dict = {k[17:] : v for k , v in moco_ckpt.items() if k[17:] in resnet50_pytorch.state_dict().keys()}
        #     overlap_state_dict = {k : v for k , v in moco_state_dict.items() if k in self.yolov4._Build_Model__yolov4.backbone.state_dict().keys()}
        #     print("MOCO Overlap[{}/{}]".format(overlap_state_dict.keys().__len__(), self.yolov4._Build_Model__yolov4.backbone.state_dict().keys().__len__()))
        #     ori = self.yolov4._Build_Model__yolov4.backbone.state_dict()
        #     ori.update(overlap_state_dict)
        #     self.yolov4._Build_Model__yolov4.backbone.load_state_dict(ori)
        #
        # elif self.args.transfer_type == 'simclr':
        #     print("-" * 100)
        #     print("load resnet50 [SimCLR] weight")
        #     print("-" * 100)
        #     base_weights = torch.load('./weight/simclr_weight.pt', map_location='cuda')['state_dict']
        #     overlap_state_dict = {k : v for k , v in base_weights.items() if k in self.yolov4._Build_Model__yolov4.backbone.state_dict().keys()}
        #     print("SimCLR Overlap[{}/{}]".format(overlap_state_dict.keys().__len__(), self.yolov4._Build_Model__yolov4.backbone.state_dict().keys().__len__()))
        #     self.yolov4._Build_Model__yolov4.backbone.load_state_dict(overlap_state_dict)
        else:
            assert False

        # Prune the backbone model
        if not self.args.full_model_transfer:
            if self.args.prune_method.lower() == "omp":
                print('execute OMP prune rate: {}'.format(self.args.prune_rate))
                pruning_model(self.yolov4._Build_Model__yolov4.backbone, self.args.prune_rate, conv1=self.args.conv1)
                check_sparsity(self.yolov4._Build_Model__yolov4.backbone, use_mask=True, conv1=self.args.conv1)

            else:
                print(f"Prune Method: {self.args.prune_method} | Mask DIR:[{self.args.mask_dir}]")
                mask = torch.load(self.args.mask_dir, map_location="cuda")
                self.args.mask = mask['mask']
                prune_model_custom(self.yolov4._Build_Model__yolov4.backbone, self.args.mask, conv1=self.args.conv1)
                check_sparsity(self.yolov4._Build_Model__yolov4.backbone, use_mask=True, conv1=self.args.conv1)

            # mask_dict = torch.load(self.args.mask_dir, map_location="cuda")
            # pruning.imp_pruning_yolo_resnet50(self.yolov4._Build_Model__yolov4.backbone, mask_dict)
            # pruning.see_zero_rate(self.yolov4._Build_Model__yolov4.backbone)
            print("-" * 100)
            print("INFO: Finish Process!")
            print("INFO: Begin Training Model")
            print('-' * 100)

        if self.args.resume_all:
            self.start_epoch = pruning.resume_begin(self.yolov4, self.optimizer, str(self.args.ckpt_base_dir))
            logger.info(" =======  Resume  Training at Epoch:[{}]  ======".format(self.start_epoch))
        else:
            logger.info(" =======  Start  Training   ======")

        start = time.time()
        for epoch in range(self.start_epoch, self.epochs):

            self.yolov4.train()
            mloss = torch.zeros(4)

            for i, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) in enumerate(
                    self.train_dataloader):
                self.scheduler.step(
                    len(self.train_dataloader)
                    / (cfg.TRAIN["BATCH_SIZE"])
                    * epoch
                    + i
                )
                imgs = imgs.to(self.device)
                label_sbbox = label_sbbox.to(self.device)
                label_mbbox = label_mbbox.to(self.device)
                label_lbbox = label_lbbox.to(self.device)
                sbboxes = sbboxes.to(self.device)
                mbboxes = mbboxes.to(self.device)
                lbboxes = lbboxes.to(self.device)

                p, p_d = self.yolov4(imgs)
                loss, loss_ciou, loss_conf, loss_cls = self.criterion(
                    p,
                    p_d,
                    label_sbbox,
                    label_mbbox,
                    label_lbbox,
                    sbboxes,
                    mbboxes,
                    lbboxes,
                )

                if self.fp_16:
                    pass
                    # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    #     scaled_loss.backward()
                else:
                    loss.backward()
                # Accumulate gradient for x batches before optimizing
                if i % self.accumulate == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Update running mean of tracked metrics
                loss_items = torch.tensor(
                    [loss_ciou, loss_conf, loss_cls, loss]
                )
                mloss = (mloss * i + loss_items) / (i + 1)

                # Print batch results
                if i % 10 == 0:
                    logger.info(
                        "  === Epoch:[{:3}/{}],step:[{:3}/{}],img_size:[{:3}],total_loss:{:.4f}|loss_ciou:{:.4f}|loss_conf:{:.4f}|loss_cls:{:.4f}|lr:{}".format(
                            epoch,
                            self.epochs,
                            i,
                            len(self.train_dataloader) - 1,
                            self.train_dataset.img_size,
                            mloss[3],
                            mloss[0],
                            mloss[1],
                            mloss[2],
                            self.optimizer.param_groups[0]["lr"],
                        )
                    )
                # multi-sclae training (320-608 pixels) every 10 batches
                if self.multi_scale_train and (i + 1) % 10 == 0:
                    self.train_dataset.img_size = (
                            random.choice(range(10, 20)) * 32
                    )
                # break
            pruning.resume_save_epoch(epoch, self.yolov4, self.optimizer, path=str(self.args.ckpt_base_dir))
            # if epoch == 5: break

        print("=========EVAL=========")
        with torch.no_grad():
            APs_all, inference_time = Evaluator(self.yolov4, showatt=False,
                                                result_path=str(self.args.ckpt_base_dir)).APs_voc()
            ap_final, ap50, ap75 = ap.compute_all_aps(APs_all, self.train_dataset.num_classes)
            logger.info("AP:[{}] AP50:[{}] AP75:[{}]".format(ap_final, ap50, ap75))
            logger.info("inference time: {:.2f} ms".format(inference_time))

        end = time.time()
        logger.info("  ===cost time:{:.4f}s".format(end - start))
        logger.info("eps {} AP:[{:.4f}] AP50:[{:.4f}] AP75:[{:.4f}]".format(self.args.eps, ap_final, ap50, ap75))

        sp = str(self.args.prune_rate) if self.args.prune_method == "omp" else \
            self.args.mask_dir.split('/')[-1].split('_')[-1][:-len('.pth.tar')]
        outp_str = (
                       "nat" if self.args.pytorch_pretrained else f"adv{self.args.eps}") + self.args.name + " " + self.args.prune_method + " " + sp + f" AP={ap_final:.4f} \n"
        print(outp_str)
        file_name = f"result_log.txt"
        f = open(file_name, "a+")
        f.write(outp_str)
        f.close()

        print('-' * 100)
        print("Finish Training")
        # save_name = './' + OUTDIR + '/transfer_{}.pth'.format(self.args.eps)
        # torch.save(self.yolov4.state_dict(), save_name)
        # print("Save: [{}]".format(save_name))
        print('-' * 100)


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


if __name__ == "__main__":
    global logger, writer
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_all", action="store_true", default=False, help="resume training flag")
    parser.add_argument('--transfer_type', default='imagenet', type=str, help='imagenet, moco, simclr')
    parser.add_argument('--prune_method', default='imp', type=str, help='omp, omp_structured, imp, rst')
    parser.add_argument('--mask_dir',
                        default='/home/yuanye/RST/Detection/tickets/R50_Linf_Eps2/mask_state2_sp51.2.pth.tar',
                        type=str, help='mask directory')
    parser.add_argument('--prune_rate', default=0.2, type=float)
    parser.add_argument('--full_model_transfer', action='store_true', default=False,
                        help="full model transfer baseline")
    parser.add_argument('--pytorch_pretrained', action='store_true', default=False)
    parser.add_argument('--eps', default=0, type=float, help='mask number')
    parser.add_argument('--seed', default=142, type=int, help='Seed')
    parser.add_argument('--log_dir', default='runs', type=str)
    parser.add_argument('--name', default='debug_runs', type=str, help='experiment name')
    parser.add_argument(
        "--weight_path",
        type=str,
        # default=None,
        default="/home/yuanye/RST/imp/pretrained_models/resnet50_l2_eps0.01.ckpt",
        help="weight file path",
    )  # weight/darknet53_448.weights
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume training flag",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="whither use GPU(0) or CPU(-1)",
    )
    parser.add_argument(
        "--accumulate",
        type=int,
        default=2,
        help="batches to accumulate before optimizing",
    )
    parser.add_argument(
        "--fp_16",
        type=bool,
        default=False,
        help="whither to use fp16 precision",
    )
    parser.add_argument('--conv1', action='store_true', help="if true, prune the conv1, else skip it")

    opt = parser.parse_args()

    os.makedirs(opt.log_dir, exist_ok=True)
    opt.run_base_dir, opt.ckpt_base_dir, opt.log_base_dir = get_directories(opt)
    os.makedirs(opt.ckpt_base_dir, exist_ok=True)

    writer = SummaryWriter(logdir=str(opt.log_base_dir))
    logger = Logger(
        log_file_name=str(opt.run_base_dir) + "/log.txt",
        log_level=logging.DEBUG,
        logger_name="YOLOv4",
    ).get_log()

    Trainer(
        weight_path=opt.weight_path,
        resume=opt.resume,
        gpu_id=opt.gpu_id,
        accumulate=opt.accumulate,
        fp_16=opt.fp_16,
        args=opt

    ).train()
