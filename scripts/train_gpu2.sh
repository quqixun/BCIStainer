#!/bin/bash


trainer=basic
config_file_list=(
    # ./configs/style_translator/exp4.yaml
    # ./configs/style_translator/exp5.yaml
    # ./configs/style_translator/exp6.yaml
    # ./configs/style_translator_cmp/exp1.yaml
    # ./configs/style_translator_cmp/exp2.yaml
    # ./configs/style_translator_cmp/exp3.yaml
    # ./configs/style_translator_cmp/exp4.yaml
)

for config_file in ${config_file_list[@]}; do

    CUDA_VISIBLE_DEVICES=2          \
    python train.py                 \
        --train_dir   ./data/train  \
        --val_dir     ./data/val    \
        --exp_root    ./experiments \
        --config_file $config_file  \
        --trainer     $trainer

done


trainer=cahr
config_file_list=(
    # ./configs/style_translator_cahr/exp6.yaml
    # ./configs/style_translator_cahr/exp9.yaml
    # ./configs/style_translator_cahr/exp13.yaml
    # ./configs/style_translator_cahr/exp14.yaml
    # ./configs/style_translator_cahr_cmp/exp1.yaml
    # ./configs/style_translator_cahr_cmp/exp2.yaml
    # ./configs/style_translator_cahr_cmp/exp3.yaml
    # ./configs/style_translator_cahr_cmp/exp4.yaml
    # ./configs/style_translator_cahr_cmp/exp5.yaml
    # ./configs/style_translator_cahr_cmp/exp6.yaml
    # ./configs/style_translator_cahr_cmp/exp7.yaml
    # ./configs/style_translator_cahr_cmp/exp8.yaml
    ./configs/style_translator_cahr/exp15.yaml
    ./configs/style_translator_cahr/exp16.yaml
    ./configs/style_translator_cahr_cmp/exp11.yaml
    ./configs/style_translator_cahr_cmp/exp12.yaml
)

for config_file in ${config_file_list[@]}; do

    CUDA_VISIBLE_DEVICES=2          \
    python train.py                 \
        --train_dir   ./data/train  \
        --val_dir     ./data/val    \
        --exp_root    ./experiments \
        --config_file $config_file  \
        --trainer     $trainer

done
