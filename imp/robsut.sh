export MIOPEN_USER_DB_PATH=/scratch/yf22/ 
#HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_imp.py
#HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_imp.py --attack_type None --pytorch-pretrained --model-path ''
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_imp.py  --model-path '' --resume 'runs/R50_Linf_Eps2_2/checkpoints/checkpoint.pth.tar'

# Downstream
CUDA_VISIBLE_DEVICES=1 python main_imp.py --attack_type None
