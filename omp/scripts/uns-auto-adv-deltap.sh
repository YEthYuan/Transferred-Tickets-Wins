#!/bin/bash

p=0
until [ ! $p -lt 10 ]
do
  echo -e "\n\n"
  echo "Now Pruning Rate ${p}%"
  echo -e "\n\n"
  sleep 1s

  python main.py \
    --arch resnet18 \
    --eps 3 \
    --prune_percent ${p} \
    --dataset cifar10 \
    --exp-name img-c10-resnet18-eps3-uns-p${p}-finetune \
    --epochs 150 \
    --opt sgd \
    --lr 0.001 \
    --step-lr 50 \
    --batch-size 64 \
    --weight-decay 5e-4 \
    --adv-train 0 \
    --adv-eval 0 \
    --workers 16 \
    --model-path pretrained_models/resnet18_l2_eps3.ckpt \
    --conv1 \
    --freeze-level -1

  p=`expr $p + 10`

done
