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
    --dataset svhn \
    --exp-name img-svhn-resnet18-eps1-uns-p${p}-finetune \
    --epochs 150 \
    --opt sgd \
    --lr 0.004 \
    --step-lr 50 \
    --batch-size 256 \
    --weight-decay 5e-4 \
    --adv-train 0 \
    --adv-eval 0 \
    --workers 32 \
    --model-path pretrained_models/resnet18_l2_eps1.ckpt \
    --conv1 \
    --freeze-level -1

  p=`expr $p + 10`

done
