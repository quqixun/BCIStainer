#!/bin/bash


config_file_list=(
    ./configs/resnet_ada_l/exp1.yaml
    ./configs/resnet_ada_l/exp2.yaml
    ./configs/resnet_ada_l/exp3.yaml
    ./configs/resnet_ada_l/exp4.yaml
)


for config_file in ${config_file_list[@]}; do
    # echo $config_file

    CUDA_VISIBLE_DEVICES=0                    \
    python train.py                           \
        --train_dir   ./data/train            \
        --val_dir     ./data/val              \
        --exp_root    ./experiments           \
        --config_file $config_file

done
