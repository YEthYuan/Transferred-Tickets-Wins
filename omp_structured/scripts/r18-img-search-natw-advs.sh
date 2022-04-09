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
    --attack_type pgd \
    --prune_percent ${p} \
    --task search \
    --set ImageNet \
    --data /home/yuanye/data \
    --pretrained -------eps=0_weight_path------------ \
    --name img-nat_weight-adv_search \
    --config config_rst/resnet18-ukn-unsigned-imagenet.yaml \
    --epochs 90 \
    --optimizer sgd \
    --lr 0.256 \
    --lr_policy multistep_lr_imagenet \
    --warmup_length 5 \
    --batch-size 256 \
    --weight-decay 0.000030517578125 \
    --momentum 0.875 \
    --label_smoothing 0.1 \
    --workers 32 \
    --epsilon 3 \
    --alpha 2 \
    --attack_iters 2

  p=`expr $p + 10`

done
