import os
import time
import thread
import logging

pydir = r"/home/sw99/.conda/envs/pytorch/bin/python"
arch = "resnet50"
cuda = 0  # "deltacuda"
run_mode = 3  # 0: 0.1~0.9  1: 0.91~0.99  2: all rate  3: only zero
adv_nat = 1  # 0: only nat  1: only adv  2: adv+nat
# ===========
log = logging.getLogger(__name__)
log_path = "launching_log.txt"
handlers = [logging.FileHandler(log_path, mode='a+'), logging.StreamHandler()]
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    handlers=handlers)
rl0 = [x / 10 for x in range(1, 10, 1)]
rl1 = [x / 100 for x in range(91, 100, 1)]
rl = [rl0, rl1, rl0 + rl1, [0]]

for adv_eps in [0, 0.01, 0.03, 0.05, 0.1, 0.25, 0.5, 1, 3, 5]:
    dadv = {"eps": adv_eps}
    dnat = {"eps": 0}
    ladv = [dadv]
    lnat = [dnat]
    ll = [lnat, ladv, lnat + ladv]
    arch_short = "r18" if arch == "resnet18" else "r50"

    for local_dict in ll[adv_nat]:
        eps = local_dict["eps"]
        eps_layout = f"eps{eps}" if eps != 0 else "nat"
        inner = ""
        if eps == 0:
            inner = "--pytorch-pretrained"
        else:
            inner = f"--model-path pretrained_models/{arch}_l2_eps{eps}.ckpt"

        for rate in rl[run_mode]:
            start = time.time()
            exp_name = f"{ds_short}-{arch_short}-{eps_layout}-{pr_method}-p{rate}-{task}"

            shell_code = f"CUDA_VISIBLE_DEVICES={cuda} {pydir} main.py --arch {arch} --eps {eps} --prune_rate {rate} --dataset {ds} --exp-name {exp_name} --epochs 150 --opt sgd --lr {lr} --step-lr 50 --batch-size 64 --weight-decay 5e-4 --adv-train 0 --adv-eval 0 --workers 32 {inner} --conv1 --freeze-level {fl}"
            log.info(f"Ready to execute: {shell_code}")
            os.system(shell_code)

            end = time.time()
            seconds = end - start
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            log.info(f"{exp_name} Duration: " + "%02d:%02d:%02d" % (h, m, s))
