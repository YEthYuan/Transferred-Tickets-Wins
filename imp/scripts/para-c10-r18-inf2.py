import math
import os
import re
import _thread
import time
import logging
import argparse
parser = argparse.ArgumentParser(description='Launch script')
args = parser.parse_args()

args.pydir = r"/home/yuanye/anaconda3/envs/t38/bin/python"
args.dir = r"/home/yuanye/RST/imp/tickets/R18_c10_Linf_Eps2"
args.ignore = [
    # put the state num of ignored masks here. eg. 0,1,4,5 (ignore state 0,1,4,5 masks)
]
args.arch = "resnet18"
args.ds = "cifar10"
args.l_cuda = [0, 1, 2, 3, 4, 5, 6, 7]  # [0,1,2,3,4,5,6,7]
# ===========
args.init_weight = ""
args.mask_list = []
args.log = logging.getLogger(__name__)
args.log_path = "runtime_log.txt"
handlers = [logging.FileHandler(args.log_path, mode='a+'), logging.StreamHandler()]
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    handlers=handlers)
args.files = os.walk(dir)
for path, dir_list, file_list in args.files:
    for file_name in file_list:
        if 'weight_init.pth.tar' in file_name:
            init_weight = os.path.join(path, file_name)
        elif 'mask' in file_name:
            if int(re.findall(r"\d+", file_name)[0]) in ignore:
                log.info(f"ignore mask {file_name}")
                continue
            else:
                mask_list.append(os.path.join(path, file_name))

num_mask = len(mask_list)
num_cuda = len(l_cuda)
mask_per_cuda = math.ceil(num_mask / num_cuda)
mask_list.sort()
log.info(f"Detected initial weight: {init_weight}")
log.info(f"Detected mask list: {mask_list}")
thread_mask = []
for i in range(0, num_mask, mask_per_cuda):
    thread_mask.append(mask_list[i:i + mask_per_cuda])
for i, masks in enumerate(thread_mask):
    log.info(f"Thread {i}: {masks}")
fn_head = dir.split('/')[-1]
fn_ds = "c10" if ds == "cifar10" else "c100" if ds == "cifar100" else ds


def train(tid, masks):
    log.info(f"Thread-{tid} launched, mask list: {masks}")
    for mask in mask_list:
        if mask in ignore:
            continue
        print("Now using mask: ", mask)
        start = time.time()
        sp = mask.split('/')[-1].split('_')[-1][:-len('.pth.tar')]
        shell_code = f"HIP_VISIBLE_DEVICES={tid} {pydir} main_eval_downstream.py --arch {arch} --set {ds} --epochs 150 --batch-size 64 --lr 0.001 --name {fn_head + '_' + sp + '_' + fn_ds} --weight_dir {init_weight} --mask_dir {mask} --trainer tune --attack_type None --workers 32 --momentum 0.9 --wd 5e-4 --save-model"
        log.info(shell_code)
        # os.system(shell_code)
        end = time.time()
        seconds = end - start
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        log.info(f"{fn_head + '_' + sp + '_' + fn_ds} Duration: " + "%02d:%02d:%02d" % (h, m, s))


for tid, masks in enumerate(thread_mask):
    try:
        _thread.start_new_thread(train, (tid, masks))
    except:
        log.info("Unable to launch Thread {}".format(tid))

while 1:
    pass