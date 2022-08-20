#!/bin/bash


config_file_list=(
    ./configs/style_translator2/exp1.yaml
    ./configs/style_translator2/exp2.yaml
    ./configs/style_translator2/exp3.yaml
    ./configs/style_translator2/exp4.yaml
    # ./configs/style_translator/exp5.yaml
    # ./configs/style_translator/exp6.yaml
    # ./configs/style_translator/exp7.yaml
    # ./configs/style_translator/exp8.yaml
)


for config_file in ${config_file_list[@]}; do

    CUDA_VISIBLE_DEVICES=0            \
    python evaluate.py                \
        --data_dir    ./data/test     \
        --exp_root    ./experiments   \
        --output_root ./evaluations   \
        --config_file $config_file    \
        --model_name  model_best_psnr \
        --apply_tta

done
