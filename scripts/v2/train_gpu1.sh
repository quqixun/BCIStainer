#!/bin/bash


config_file_list=(
    ./configs/resnet_mod_v2/exp1.yaml
    ./configs/resnet_mod_v2/exp2.yaml
    ./configs/resnet_mod_v2/exp3.yaml
)


for config_file in ${config_file_list[@]}; do
    # echo $config_file

    CUDA_VISIBLE_DEVICES=1                    \
    python train_v2.py                        \
        --train_dir   ./data/train            \
        --val_dir     ./data/val              \
        --exp_root    ./experiments           \
        --config_file $config_file

done
