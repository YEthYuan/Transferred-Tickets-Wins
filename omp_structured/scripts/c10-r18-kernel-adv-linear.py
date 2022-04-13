import os
import time
import logging

pydir = r"/home/yuanye/anaconda3/envs/t38/bin/python"
cuda = "1"  # "0,1,2"
arch = "ResNet18"
ds = "cifar10"
task = "linear"  # linear, finetune
lr = 0.01
adv_eps = 3
pr_method = "kernel"  # row, kernel, filter
run_mode = 2  # 0: 0.1~0.9  1: 0.91~0.99  2: all rate  3: only zero  4: custom rate
custom_rate = [x / 100 for x in range(91, 100, 1)]
adv_nat = 1  # 0: only nat  1: only adv  2: adv+nat
# ===========
log = logging.getLogger(__name__)
log_path = "runtime_log.txt"
handlers = [logging.FileHandler(log_path, mode='a+'), logging.StreamHandler()]
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    handlers=handlers)
rl0 = [x / 10 for x in range(1, 10, 1)]
rl1 = [x / 100 for x in range(91, 100, 1)]
rl = [rl0, rl1, rl0 + rl1, [0], custom_rate]
dadv = {"eps": adv_eps}
dnat = {"eps": 0}
ladv = [dadv]
lnat = [dnat]
ll = [lnat, ladv, lnat + ladv]
arch_short = "r18" if arch == "ResNet18" else "r50"
ds_short = ""
if ds == "cifar10":
    ds_short = "c10"
elif ds == "cifar100":
    ds_short = "c100"
elif ds == "ImageNet":
    ds_short = "img"
elif ds == "caltech101":
    ds_short = "cal101"
else:
    ds_short = ds
fl = 4 if task == "linear" else -1
if pr_method == "filter":
    ct = "SubnetConv_filter"
elif pr_method == "kernel":
    ct = "SubnetConv_kernel"
elif pr_method == "row":
    ct = "SubnetConv_row"
else:
    ct = None
    raise Exception(f"Not support the prune method {pr_method}")

for local_dict in ll[adv_nat]:
    eps = local_dict["eps"]
    eps_layout = f"eps{eps}" if eps != 0 else "nat"
    inner = ""
    if eps == 0:
        inner = "--pytorch-pretrained"
    else:
        inner = f"--pretrained pretrained_models/{arch.lower()}_l2_eps{eps}.ckpt"

    for rate in rl[run_mode]:
        start = time.time()
        exp_name = f"{ds_short}-{arch_short}-{eps_layout}-{pr_method}-p{rate}-{task}"

        shell_code = f"CUDA_VISIBLE_DEVICES={cuda} {pydir} main.py --arch {arch} --epsilon {eps} --conv_type {ct} --prune_rate {rate} --set {ds} --name {exp_name} --epochs 150 --opt sgd --lr {lr} --lr_policy multistep_lr --batch_size 64 --weight-decay 5e-4 --workers 32 {inner} --freeze-level {fl}"
        log.info(f"Ready to execute: {shell_code}")
        os.system(shell_code)

        end = time.time()
        seconds = end - start
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        log.info(f"{exp_name} Duration: " + "%02d:%02d:%02d" % (h, m, s))
