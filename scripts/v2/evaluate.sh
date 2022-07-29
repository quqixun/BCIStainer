#!/bin/bash


config_file_list=(
    ./configs/resnet_ada_v2/exp1.yaml
    ./configs/resnet_ada_v2/exp2.yaml
    ./configs/resnet_ada_v2/exp3.yaml
    # ./configs/resnet_ada_v2/exp4.yaml
)


for config_file in ${config_file_list[@]}; do

    CUDA_VISIBLE_DEVICES=0            \
    python evaluate.py                \
        --data_dir    ./data/val      \
        --exp_root    ./experiments   \
        --output_root ./outputs       \
        --config_file $config_file    \
        --model_name  model_best_psnr \
        --apply_tta

done
