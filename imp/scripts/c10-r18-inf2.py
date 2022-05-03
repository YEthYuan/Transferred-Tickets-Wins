import os
import argparse
import re
import time
import logging

parser = argparse.ArgumentParser(description='launch script')
parser.add_argument('--mask', metavar='N', type=int, nargs='+', default=[1, 2],  # default=[i for i in range(19)],
                    help='--mask 0 1 4 5 (run state 0,1,4,5 masks on this cuda)')
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--workers', type=int, default=32)
parser.add_argument('--pydir', type=str, default=r"/home/yuanye/anaconda3/envs/t38/bin/python", help='python directory')
parser.add_argument('--dir', type=str, default=r"/home/yuanye/RST/imp/tickets/R18_c10_Linf_Eps2", help='mask directory')
parser.add_argument('--arch', type=str, default="resnet18", help='resnet18 , resnet50')
parser.add_argument('--ds', type=str, default="cifar10", help='cifar10 cifar100')
args = parser.parse_args()
print(args)
# ===========
log = logging.getLogger(__name__)
log_path = "runtime_log.txt"
handlers = [logging.FileHandler(log_path, mode='a+'), logging.StreamHandler()]
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    handlers=handlers)
log.info(args)
init_weight = ""
mask_list = []
files = os.walk(args.dir)
for path, dir_list, file_list in files:
    for file_name in file_list:
        if 'weight_init.pth.tar' in file_name:
            init_weight = os.path.join(path, file_name)
        elif 'mask' in file_name and int(re.findall(r"\d+", file_name)[0]) in args.mask:
            mask_list.append(os.path.join(path, file_name))

mask_list.sort()
log.info(f"CUDA {args.cuda} mask:{mask_list}")
fn_head = args.dir.split('/')[-1]
fn_ds = "c10" if args.ds == "cifar10" else "c100" if args.ds == "cifar100" else args.ds
log.info("auto-detected init_weight_path: ", init_weight)
for mask in mask_list:
    start = time.time()
    log.info(f"Now using mask: {mask}")
    sp = mask.split('/')[-1].split('_')[-1][:-len('.pth.tar')]
    shell_code = f"HIP_VISIBLE_DEVICES={args.cuda} {args.pydir} main_eval_downstream.py --arch {args.arch} --set {args.ds} --epochs 150 --batch-size 64 --lr 0.001 --name {fn_head + '_' + sp + '_' + fn_ds} --weight_dir {init_weight} --mask_dir {mask} --trainer tune --attack_type None --workers {args.workers} --momentum 0.9 --wd 5e-4 --save-model"
    log.info(shell_code)
    os.system(shell_code)
    end = time.time()
    seconds = end - start
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    log.info(f"{fn_head + '_' + sp + '_' + fn_ds} Duration: " + "%02d:%02d:%02d" % (h, m, s))
