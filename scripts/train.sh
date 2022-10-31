#!/bin/bash


# settings
device=0
trainer=basic

# configurations of experiment
config_file=./configs/stainer_basic_cmp/exp3.yaml

# training
CUDA_VISIBLE_DEVICES=$device    \
python train.py                 \
    --train_dir   ./data/train  \
    --val_dir     ./data/val    \
    --exp_root    ./experiments \
    --config_file $config_file  \
    --trainer     $trainer
