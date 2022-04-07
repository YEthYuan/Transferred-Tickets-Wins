#!/bin/bash

p=0
until [ ! $p -lt 100 ]
do
  echo -e "\n\n"
  echo "Now Pruning Rate ${p}%"
  echo -e "\n\n"
  sleep 1s

  python image_inversion.py \
    --prune_percent ${p} \
    --seed 66
    
  p=`expr $p + 10`

done
