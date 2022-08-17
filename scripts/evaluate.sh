#!/bin/bash


config_file_list=(
    ./configs/style_translator/exp1.yaml
    ./configs/style_translator/exp2.yaml
    ./configs/style_translator/exp3.yaml
    ./configs/style_translator/exp4.yaml
)


for config_file in ${config_file_list[@]}; do

    CUDA_VISIBLE_DEVICES=0            \
    python evaluate.py                \
        --data_dir    ./data/test     \
        --exp_root    ./experiments   \
        --output_root ./outputs_test  \
        --config_file $config_file    \
        --model_name  model_best_psnr \
        --apply_tta

done
