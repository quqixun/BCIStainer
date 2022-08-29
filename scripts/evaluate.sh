#!/bin/bash


# ------------------------------------------------------------------------------
# basic evaluator

evaluator=basic
config_file_list=(
    # ./configs/style_translator/exp5.yaml
    # ./configs/style_translator/exp6.yaml
    ./configs/style_translator/exp7.yaml
    ./configs/style_translator/exp8.yaml
)
apply_tta_list=(false true)

for config_file in ${config_file_list[@]}; do
for apply_tta in ${apply_tta_list[@]}; do

    CUDA_VISIBLE_DEVICES=0            \
    python evaluate.py                \
        --data_dir    ./data/test     \
        --exp_root    ./experiments   \
        --output_root ./evaluations   \
        --config_file $config_file    \
        --model_name  model_best_psnr \
        --apply_tta   $apply_tta      \
        --evaluator   $evaluator

done; done


# ------------------------------------------------------------------------------
# cahr evaluator

evaluator=cahr
config_file_list=(
    ./configs/style_translator_cahr/exp5.yaml
    ./configs/style_translator_cahr/exp6.yaml
    ./configs/style_translator_cahr/exp7.yaml
    ./configs/style_translator_cahr/exp8.yaml
    ./configs/style_translator_cahr/exp9.yaml
    ./configs/style_translator_cahr/exp10.yaml
    ./configs/style_translator_cahr/exp11.yaml
    ./configs/style_translator_cahr/exp12.yaml
    # ./configs/style_translator_cahr/exp13.yaml
    # ./configs/style_translator_cahr/exp14.yaml
    # ./configs/style_translator_cahr/exp15.yaml
    # ./configs/style_translator_cahr/exp16.yaml
)
apply_tta_list=(false true)

for config_file in ${config_file_list[@]}; do
for apply_tta in ${apply_tta_list[@]}; do

    CUDA_VISIBLE_DEVICES=0            \
    python evaluate.py                \
        --data_dir    ./data/test     \
        --exp_root    ./experiments   \
        --output_root ./evaluations   \
        --config_file $config_file    \
        --model_name  model_best_psnr \
        --apply_tta   $apply_tta      \
        --evaluator   $evaluator

done; done
