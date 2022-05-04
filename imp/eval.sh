export MIOPEN_USER_DB_PATH=/scratch/yf22/
python3 eval.py --mask 2 5 --cuda $1 > nat_10_$1.txt 2>&1 &
#HIP_VISIBLE_DEVICES=4,5,6,7 python3 eval.py > eval_c100.txt 2>&1 &

