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
    --attack_type fgsm-rs \
    --prune_percent ${p} \
    --task search \
    --set ImageNet \
    --data /scratch/cl114/ILSVRC/Data/CLS-LOC/ \
    --pretrained /home/yf22/ResNet_ckpt/resnet18_linf_eps2.0.ckpt \
    --name img-Linf2_weight-adv_search \
    --config config_rst/resnet18-ukn-unsigned-imagenet.yaml \
    --epochs 90 \
    --optimizer sgd \
    --lr 0.256 \
    --lr_policy multistep_lr_imagenet \
    --warmup_length 5 \
    --batch_size 256 \
    --weight-decay 0.000030517578125 \
    --momentum 0.875 \
    --label-smoothing 0.1 \
    --workers 32 \
    --epsilon 2 \
    --alpha 2.5 \
    --attack_iters 1 \
    --constraint Linf
  p=`expr $p + 10`

done
