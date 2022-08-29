#!/bin/bash


config_file_list=(
    ./configs/style_translator/exp5.yaml
    ./configs/style_translator/exp6.yaml
)

for config_file in ${config_file_list[@]}; do

    CUDA_VISIBLE_DEVICES=1          \
    python train.py                 \
        --train_dir   ./data/train  \
        --val_dir     ./data/val    \
        --exp_root    ./experiments \
        --config_file $config_file  \
        --trainer     basic

done


# config_file_list=(
#     ./configs/style_translator_cahr/exp10.yaml
#     ./configs/style_translator_cahr/exp14.yaml
# )

# for config_file in ${config_file_list[@]}; do

#     CUDA_VISIBLE_DEVICES=1          \
#     python train.py                 \
#         --train_dir   ./data/train  \
#         --val_dir     ./data/val    \
#         --exp_root    ./experiments \
#         --config_file $config_file  \
#         --trainer     cahr

# done
