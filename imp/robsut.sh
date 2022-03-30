export MIOPEN_USER_DB_PATH=/scratch/yf22/ 
#HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_imp.py
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_imp.py --attack_type None --pytorch-pretrained --model-path ''
