#!/bin/bash

p=10
until [ ! $p -lt 100 ]
do
  echo -e "\n\n"
  echo "Now Pruning Rate ${p}%"
  echo -e "\n\n"
  sleep 1s

  python main.py \
    --arch resnet18 \
    --eps 1 \
    --prune_percent ${p} \
    --dataset cifar100 \
    --exp-name img-c100-resnet18-eps0-uns-p${p}-finetune \
    --epochs 150 \
    --opt sgd \
    --lr 0.001 \
    --step-lr 50 \
    --batch-size 64 \
    --weight-decay 5e-4 \
    --adv-train 0 \
    --adv-eval 0 \
    --workers 0 \
    --pytorch-pretrained \
    --freeze-level -1

  p=`expr $p + 10`

done
