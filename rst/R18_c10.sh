#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python main.py \
  --arch ResNet18 \
  --attack_type None \
  --prune-rate $2 \
  --task search \
  --set flowers \
  --data /home/sw99/datasets \
  --pytorch-pretrained \
  --name flowers_search \
  --config config_rst/resnet18-ukn-unsigned-imagenet.yaml \
  --conv_type SubnetConv \
  --epochs 160 \
  --optimizer sgd \
  --lr 0.1 \
  --lr_policy cifar_piecewise \
  --batch_size 256 \
  --weight-decay 0.0005 \
  --momentum 0.9 \
  --workers 32 \
  --epsilon 3 \
  --alpha 10 \
  --attack_iters 7 > flowers_$2.txt 2>&1 &

