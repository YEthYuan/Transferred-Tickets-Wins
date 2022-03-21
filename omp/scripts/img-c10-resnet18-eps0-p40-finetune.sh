python main.py \
  --arch resnet18 \
  --eps 3 \
  --prune_rate 0.4 \
  --dataset cifar10 \
  --exp-name img-c10-resnet18-eps0-p40-finetune \
  --epochs 150 \
  --lr 0.01 \
  --step-lr 30 \
  --batch-size 64 \
  --weight-decay 5e-4 \
  --adv-train 0 \
  --adv-eval 1 \
  --workers 0 \
  --pytorch-pretrained \
  --freeze-level -1