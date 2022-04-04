#export MIOPEN_USER_DB_PATH=/scratch/yf22/ 
#HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_imp.py
#HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_imp.py --attack_type None --pytorch-pretrained --model-path ''
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_imp.py  --model-path '' --resume 'runs/R50_Linf_Eps2_2/checkpoints/checkpoint.pth.tar'

# Downstream
CUDA_VISIBLE_DEVICES=$1 python main_imp.py --attack_type None > train_$1.txt 2>&1 &

# Cifar10
# HIP_VISIBLE_DEVICES=0 python main_imp.py --arch 'resnet18' --set 'cifar10' --model-path '/home/yf22/ResNet_ckpt/resnet50_linf_eps4.0.ckpt' --name 'R18_c10_Linf_Eps2'
HIP_VISIBLE_DEVICES=0,1 python main_imp.py --arch 'resnet18' --set 'cifar10' --model-path '' --resume 'runs/R18_c10_Linf_Eps2/checkpoints/checkpoint.pth.tar' --name 'R18_c10_Linf_Eps2' > R18_2_c10.txt 2>&1 &
HIP_VISIBLE_DEVICES=2,3 python main_imp.py --arch 'resnet18' --set 'cifar10' --model-path '' --resume 'runs/R18_c10_Linf_Eps4/checkpoints/checkpoint.pth.tar' --name 'R18_c10_Linf_Eps4' > R18_4_c10.txt 2>&1 &


# Cifar100
HIP_VISIBLE_DEVICES=4,5 python main_imp.py --arch 'resnet18' --set 'cifar100' --model-path '' --resume 'runs/R18_c100_Linf_Eps2/checkpoints/checkpoint.pth.tar' --name 'R18_c100_Linf_Eps2' > R18_2_c100.txt 2>&1 &
HIP_VISIBLE_DEVICES=6,7 python main_imp.py --arch 'resnet18' --set 'cifar100' --model-path '' --resume 'runs/R18_c100_Linf_Eps4/checkpoints/checkpoint.pth.tar' --name 'R18_c100_Linf_Eps4' > R18_4_c100.txt 2>&1 &