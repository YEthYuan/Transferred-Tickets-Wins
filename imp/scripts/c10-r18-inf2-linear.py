import os
pydir = r"/home/yuanye/anaconda3/envs/t38/bin/python"
dir = r"/home/yuanye/RST/imp/tickets/R18_inf2"
ignore = [

]
arch = "resnet18"
ds = "cifar10"
# ===========
init_weight = ""
mask_list = []
files = os.walk(dir)
for path, dir_list, file_list in files:
    for file_name in file_list:
        if 'weight_init.pth.tar' in file_name:
            init_weight = os.path.join(path, file_name)
        elif 'mask' in file_name:
            mask_list.append(os.path.join(path, file_name))

mask_list.sort()
fn_head = dir.split('/')[-1]
fn_ds = "c10" if ds is "cifar10" else "c100" if ds is "cifar100" else ds
print("auto-detected init_weight_path: ", init_weight)
for mask in mask_list:
    if mask in ignore:
        continue
    print("Now using mask: ", mask)
    sp = mask.split('/')[-1].split('_')[-1][:-len('.pth.tar')]
    shell_code = f"{pydir} main_eval_downstream.py --arch {arch} --set {ds} --epochs 150 --batch-size 64 --lr 0.01 --name {'linear_'+fn_head+'_'+sp+'_'+fn_ds} --weight_dir {init_weight} --mask_dir {mask} --trainer tune --attack_type None --workers 32 --momentum 0.9 --wd 5e-4 --save-model --linear-eval"
    print("Ready to execute: ", shell_code)
    os.system(shell_code)
