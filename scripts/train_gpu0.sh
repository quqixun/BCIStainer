#!/bin/bash


config_file_list=(
    ./configs/resnet_ada/exp12.yaml
    ./configs/resnet_ada/exp13.yaml
    ./configs/resnet_ada_l/exp1.yaml
    ./configs/resnet_ada_h/exp1.yaml
    ./configs/unet_ada/exp1.yaml
    ./configs/resnet_ada_l/exp2.yaml
    ./configs/resnet_ada_h/exp2.yaml
    ./configs/unet_ada/exp2.yaml
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
