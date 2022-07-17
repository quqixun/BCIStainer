#!/bin/bash


config_file_list=(
    # ./configs/baseline/exp1.yaml
    # ./configs/baseline/exp2.yaml
    # ./configs/baseline/exp3.yaml
    ./configs/baseline/exp4.yaml
    ./configs/baseline/exp5.yaml
    # ./configs/baseline/exp6.yaml
    # ./configs/unet/exp1.yaml
    # ./configs/unet/exp2.yaml
    # ./configs/unet/exp3.yaml
    # ./configs/unet/exp4.yaml
    # ./configs/unet/exp5.yaml
    # ./configs/unet/exp6.yaml
    # ./configs/unet/exp7.yaml
    # ./configs/unet/exp8.yaml
    # ./configs/unet/exp9.yaml
    # ./configs/unet/exp10.yaml
    # ./configs/unet/exp11.yaml
    # ./configs/unet/exp12.yaml
    ./configs/unet/exp13.yaml
    ./configs/unet/exp14.yaml
    ./configs/unet/exp15.yaml
    ./configs/unet/exp16.yaml
    ./configs/unet/exp17.yaml
    ./configs/unet/exp18.yaml
)

model_name_list=(
    model_best_psnr
    # model_best_ssim
    # model_latest
)


for config_file in ${config_file_list[@]}; do
for model_name in ${model_name_list[@]}; do

    CUDA_VISIBLE_DEVICES=0          \
    python evaluate.py              \
        --data_dir    ./data/val    \
        --exp_root    ./experiments \
        --output_root ./outputs     \
        --config_file $config_file  \
        --model_name  $model_name   \
        --apply_tta

done; done
