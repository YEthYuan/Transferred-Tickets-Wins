#!/bin/bash

p=10
until [ ! $p -lt 100 ]
do
  echo -e "\n\n"
  echo "Now Pruning Rate ${p}%"
  echo -e "\n\n"
  sleep 1s

  python main.py \
    --arch ResNet18 \
    --attack_type None \
    --prune_percent ${p} \
    --task search \
    --set CIFAR10 \
    --data /home/yuanye/data \
    --name img-nat_weight-nat_search \
    --config config_rst/resnet18-ukn-unsigned-imagenet.yaml \
    --conv_type SubnetConv \
    --epochs 160 \
    --optimizer sgd \
    --lr 0.1 \
    --lr_policy cifar_piecewise \
    --batch-size 256 \
    --weight-decay 0.0005 \
    --momentum 0.9 \
    --workers 32 \
    --epsilon 3 \
    --alpha 10 \
    --attack_iters 7

  p=`expr $p + 10`

done
