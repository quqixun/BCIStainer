#!/bin/bash


model_name=model_best_psnr
apply_tta_list=(false true)


evaluator=basic
config_file_list=(
    # ./configs/style_translator/exp4.yaml
    # ./configs/style_translator/exp5.yaml
    # ./configs/style_translator/exp6.yaml
    # ./configs/style_translator/exp7.yaml
    # ./configs/style_translator_cmp/exp1.yaml
    # ./configs/style_translator_cmp/exp2.yaml
    # ./configs/style_translator_cmp/exp3.yaml
    # ./configs/style_translator_cmp/exp4.yaml
    # ./configs/style_translator_cmp/exp5.yaml
    # ./configs/style_translator_cmp/exp6.yaml
    # ./configs/style_translator_cmp/exp7.yaml
    # ./configs/style_translator_cmp/exp8.yaml
    # ./configs/style_translator_cmp/exp9.yaml
)

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


evaluator=cahr
config_file_list=(
    # ./configs/style_translator_cahr/exp6.yaml
    # ./configs/style_translator_cahr/exp9.yaml
    # ./configs/style_translator_cahr/exp13.yaml
    # ./configs/style_translator_cahr/exp14.yaml
    # ./configs/style_translator_cahr/exp15.yaml
    # ./configs/style_translator_cahr/exp16.yaml
    # ./configs/style_translator_cahr/exp17.yaml
    # ./configs/style_translator_cahr/exp18.yaml
    # ./configs/style_translator_cahr/exp19.yaml
    # ./configs/style_translator_cahr/exp20.yaml
    # ./configs/style_translator_cahr_cmp/exp1.yaml
    # ./configs/style_translator_cahr_cmp/exp2.yaml
    # ./configs/style_translator_cahr_cmp/exp3.yaml
    # ./configs/style_translator_cahr_cmp/exp4.yaml
    # ./configs/style_translator_cahr_cmp/exp5.yaml
    # ./configs/style_translator_cahr_cmp/exp6.yaml
    # ./configs/style_translator_cahr_cmp/exp7.yaml
    # ./configs/style_translator_cahr_cmp/exp8.yaml
    # ./configs/style_translator_cahr_cmp/exp9.yaml
    # ./configs/style_translator_cahr_cmp/exp10.yaml
    # ./configs/style_translator_cahr_cmp/exp11.yaml
    # ./configs/style_translator_cahr_cmp/exp12.yaml
    # ./configs/style_translator_cahr_cmp/exp13.yaml
    ./configs/style_translator_cahr_cmp/exp14.yaml
)

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
