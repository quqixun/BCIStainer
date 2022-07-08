#!/bin/bash


config_file_list=(
    ./configs/baseline/exp1.yaml
    ./configs/baseline/exp2.yaml
    ./configs/baseline/exp3.yaml
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
