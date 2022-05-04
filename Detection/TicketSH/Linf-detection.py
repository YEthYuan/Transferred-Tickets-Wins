import os
import time
import logging

pydir = r"/home/yonggan/anaconda3/envs/pytorch/bin/python"
cuda = "1"  # "0,1,2"
arch = "ResNet50"
constraint = "Linf"
adv_eps = 0.5
run_mode = 4  # 0: 0.1~0.9  1: 0.91~0.99  2: all rate  3: only zero  4: custom rate
custom_rate = [x / 10 for x in range(1, 4, 1)]
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

for local_dict in ll[adv_nat]:
    eps = local_dict["eps"]
    eps_layout = f"{constraint.lower()}_eps{eps}" if eps != 0 else "nat"
    inner = ""
    if eps == 0:
        inner = "--pytorch-pretrained"
    else:
        if constraint == "L2":
            inner = f"--weight_path ~/ResNet_ckpt/L2/{arch.lower()}_{eps_layout}.ckpt"
        elif constraint == "Linf":
            inner = f"--weight_path ~/ResNet_ckpt/{arch.lower()}_{eps_layout}.ckpt"

    for rate in rl[run_mode]:
        start = time.time()
        exp_name = f"{eps_layout}-sp{rate}"

        shell_code = f"CUDA_VISIBLE_DEVICES={cuda} {pydir} train_transfer.py --eps {adv_eps} {inner} --name {exp_name} --prune_method omp --prune_rate {rate}"
        log.info(f"Ready to execute: {shell_code}")
        os.system(shell_code)

        end = time.time()
        seconds = end - start
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        log.info(f"{exp_name} Duration: " + "%02d:%02d:%02d" % (h, m, s))
