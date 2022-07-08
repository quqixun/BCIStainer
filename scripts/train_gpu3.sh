#!/bin/bash


config_file_list=(
    ./configs/unet/exp1.yaml
    ./configs/unet/exp2.yaml
    ./configs/unet/exp3.yaml
)


for config_file in ${config_file_list[@]}; do
    # echo $config_file

    CUDA_VISIBLE_DEVICES=3                    \
    python train.py                           \
        --train_dir   ./data/train            \
        --val_dir     ./data/val              \
        --exp_root    ./experiments           \
        --config_file $config_file

done