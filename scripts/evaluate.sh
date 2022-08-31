#!/bin/bash


# ------------------------------------------------------------------------------
# basic evaluator

evaluator=basic
config_file_list=(
    # ./configs/style_translator/exp5.yaml
    # ./configs/style_translator/exp7.yaml
    # ./configs/style_translator/exp8.yaml
    # ./configs/style_translator/exp9.yaml
    # ./configs/style_translator/exp10.yaml
    # ./configs/style_translator/exp12.yaml
    # ./configs/style_translator/exp13.yaml
    ./configs/style_translator/exp6.yaml
    ./configs/style_translator/exp11.yaml
    ./configs/style_translator/exp14.yaml
    ./configs/style_translator_cmp/exp1.yaml
    ./configs/style_translator_cmp/exp2.yaml
)
model_name_list=(model_best_psnr)  # model_best_ssim
apply_tta_list=(false true)

for config_file in ${config_file_list[@]}; do
for model_name in ${model_name_list[@]}; do
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

done; done; done


# ------------------------------------------------------------------------------
# cahr evaluator

# evaluator=cahr
# config_file_list=(
#     ./configs/style_translator_cahr/exp1.yaml
#     ./configs/style_translator_cahr/exp2.yaml
#     ./configs/style_translator_cahr/exp3.yaml
#     ./configs/style_translator_cahr/exp4.yaml
# )
# model_name_list=(model_best_psnr)  # model_best_ssim
# apply_tta_list=(false true)

# for config_file in ${config_file_list[@]}; do
# for model_name in ${model_name_list[@]}; do
# for apply_tta in ${apply_tta_list[@]}; do

#     CUDA_VISIBLE_DEVICES=0            \
#     python evaluate.py                \
#         --data_dir    ./data/test     \
#         --exp_root    ./experiments   \
#         --output_root ./evaluations   \
#         --config_file $config_file    \
#         --model_name  $model_name     \
#         --apply_tta   $apply_tta      \
#         --evaluator   $evaluator

# done; done; done
