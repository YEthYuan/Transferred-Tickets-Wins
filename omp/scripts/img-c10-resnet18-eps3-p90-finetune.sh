python main.py \
  --arch resnet18 \
  --eps 3 \
  --prune_rate 0.9 \
  --dataset cifar10 \
  --exp-name img-c10-resnet18-eps3-p90-finetune \
  --epochs 150 \
  --lr 0.001 \
  --step-lr 50 \
  --batch-size 64 \
  --weight-decay 5e-4 \
  --adv-train 0 \
  --adv-eval 0 \
  --workers 0 \
  --model-path pretrained_models/resnet18_l2_eps3.ckpt \
  --freeze-level -1