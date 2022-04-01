#!/bin/bash

p=10
until [ ! $p -lt 100 ]
do
  echo -e "\n\n"
  echo "Now Pruning Rate ${p}%"
  echo -e "\n\n"
  sleep 1s

  python main.py \
    --arch resnet50 \
    --eps 3 \
    --prune_percent ${p} \
    --dataset cifar10 \
    --exp-name img-c10-resnet50-eps0-uns-p${p}-linear \
    --epochs 150 \
    --opt sgd \
    --lr 0.01 \
    --step-lr 50 \
    --batch-size 64 \
    --weight-decay 5e-4 \
    --adv-train 0 \
    --adv-eval 0 \
    --workers 32 \
    --pytorch-pretrained \
    --conv1 \
    --freeze-level 4
    
  p=`expr $p + 10`

done
