#!/bin/bash


config_file_list=(
    ./configs/resnet_ada_v2/exp5.yaml
    ./configs/resnet_ada_v2/exp6.yaml
    ./configs/resnet_ada_v2/exp7.yaml
    ./configs/resnet_ada_v2/exp8.yaml
    ./configs/resnet_ada_v2/exp9.yaml
    ./configs/resnet_ada_v2/exp10.yaml
    ./configs/resnet_ada_v2/exp11.yaml
    ./configs/resnet_ada_v2/exp12.yaml
)


for config_file in ${config_file_list[@]}; do
    # echo $config_file

    CUDA_VISIBLE_DEVICES=0                    \
    python train_v2.py                        \
        --train_dir   ./data/train            \
        --val_dir     ./data/val              \
        --exp_root    ./experiments           \
        --config_file $config_file

done
