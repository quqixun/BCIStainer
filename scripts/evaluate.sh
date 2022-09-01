#!/bin/bash


# basic evaluator
evaluator=basic
config_file_list=(
    ./configs/style_translator/exp1.yaml
    ./configs/style_translator/exp4.yaml
    ./configs/style_translator/exp5.yaml
    ./configs/style_translator/exp6.yaml
)

# cahr evaluator
# evaluator=cahr
# config_file_list=(
#     ./configs/style_translator_cahr/exp6.yaml
#     ./configs/style_translator_cahr/exp9.yaml
# )

model_name=model_best_psnr
apply_tta_list=(false true)

for config_file in ${config_file_list[@]}; do
for apply_tta in ${apply_tta_list[@]}; do

    CUDA_VISIBLE_DEVICES=0            \
    python evaluate.py                \
        --data_dir    ./data/test     \
        --exp_root    ./experiments   \
        --output_root ./evaluations   \
        --config_file $config_file    \
        --model_name  $model_name     \
        --apply_tta   $apply_tta      \
        --evaluator   $evaluator

done; done
