#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python main.py \
  --arch ResNet18 \
  --attack_type None \
  --prune-rate $2 \
  --task search \
  --set svhn \
  --data /home/sw99/datasets \
  --pytorch-pretrained \
  --name svhn \
  --config config_rst/resnet18-ukn-unsigned-imagenet.yaml \
  --conv_type SubnetConv \
  --epochs 160 \
  --optimizer sgd \
  --lr 1.0 \
  --lr_policy cifar_piecewise \
  --batch_size 256 \
  --weight-decay 0.0005 \
  --momentum 0.9 \
  --workers 32 \
  --epsilon 3 \
  --alpha 10 \
  --attack_iters 7 > svhn_$1.txt 2>&1 &

