#!/bin/bash


config_file_list=(
    # ./configs/baseline/exp1.yaml
    ./configs/unet/exp1.yaml
    ./configs/unet/exp2.yaml
    ./configs/unet/exp3.yaml
)

model_name_list=(
    model_best_psnr
    model_best_ssim
    model_latest
)


for config_file in ${config_file_list[@]}; do
for model_name in ${model_name_list[@]}; do

    CUDA_VISIBLE_DEVICES=0                    \
    python evaluate.py                        \
        --data_dir    ./data/val              \
        --exp_root    ./experiments           \
        --output_root ./outputs               \
        --config_file $config_file            \
        --model_name  $model_name

done; done
