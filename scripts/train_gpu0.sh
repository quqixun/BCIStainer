#!/bin/bash


config_file_list=(
    # ./configs/style_translator/exp5.yaml
    # ./configs/style_translator/exp6.yaml
    # ./configs/style_translator/exp9.yaml
    # ./configs/style_translator/exp15.yaml
    # ./configs/style_translator_cmp/exp1.yaml
    # ./configs/style_translator_cmp/exp2.yaml
    ./configs/style_translator_cmp/exp3.yaml
)

for config_file in ${config_file_list[@]}; do

    CUDA_VISIBLE_DEVICES=0          \
    python train.py                 \
        --train_dir   ./data/train  \
        --val_dir     ./data/val    \
        --exp_root    ./experiments \
        --config_file $config_file  \
        --trainer     basic

done


# config_file_list=(
#     ./configs/style_translator_cahr/exp9.yaml
#     ./configs/style_translator_cahr/exp13.yaml
# )

# for config_file in ${config_file_list[@]}; do

#     CUDA_VISIBLE_DEVICES=0          \
#     python train.py                 \
#         --train_dir   ./data/train  \
#         --val_dir     ./data/val    \
#         --exp_root    ./experiments \
#         --config_file $config_file  \
#         --trainer     cahr

# done
