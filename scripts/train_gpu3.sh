#!/bin/bash


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
    # ./configs/style_translator_ocahr/exp6.yaml
    # ./configs/style_translator_ocahr/exp9.yaml
    # ./configs/style_translator_ocahr/exp13.yaml
    # ./configs/style_translator_ocahr/exp14.yaml
    # ./configs/style_translator_ocahr_cmp/exp1.yaml
    # ./configs/style_translator_ocahr_cmp/exp2.yaml
    # ./configs/style_translator_ocahr_cmp/exp3.yaml
    # ./configs/style_translator_ocahr_cmp/exp4.yaml
    # ./configs/style_translator_ocahr_cmp/exp5.yaml
    # ./configs/style_translator_ocahr_cmp/exp6.yaml
    # ./configs/style_translator_ocahr_cmp/exp7.yaml
    # ./configs/style_translator_ocahr_cmp/exp8.yaml
    ./configs/style_translator_cahr/exp17.yaml
    ./configs/style_translator_cahr/exp18.yaml
    ./configs/style_translator_cahr_cmp/exp9.yaml
    ./configs/style_translator_cahr_cmp/exp10.yaml
    ./configs/style_translator_cahr_cmp/exp11.yaml
    ./configs/style_translator_cahr_cmp/exp12.yaml
)

for config_file in ${config_file_list[@]}; do

    CUDA_VISIBLE_DEVICES=3          \
    python train.py                 \
        --train_dir   ./data/train  \
        --val_dir     ./data/val    \
        --exp_root    ./experiments \
        --config_file $config_file  \
        --trainer     $trainer

done
