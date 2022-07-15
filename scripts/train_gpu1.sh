#!/bin/bash


config_file_list=(
    ./configs/unet/exp16.yaml
    ./configs/unet/exp17.yaml
    ./configs/unet/exp18.yaml
)


for config_file in ${config_file_list[@]}; do
    # echo $config_file

    CUDA_VISIBLE_DEVICES=1                    \
    python train.py                           \
        --train_dir   ./data/train            \
        --val_dir     ./data/val              \
        --exp_root    ./experiments           \
        --config_file $config_file

done
