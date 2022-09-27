#!/bin/bash


model_name=model_best_psnr
apply_tta_list=(false true)

evaluator=cahr
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
    ./configs/style_translator_ocahr/exp6.yaml
    ./configs/style_translator_ocahr/exp9.yaml
    ./configs/style_translator_ocahr/exp13.yaml
    ./configs/style_translator_ocahr/exp14.yaml
    ./configs/style_translator_ocahr_cmp/exp1.yaml
    ./configs/style_translator_ocahr_cmp/exp2.yaml
    ./configs/style_translator_ocahr_cmp/exp3.yaml
    ./configs/style_translator_ocahr_cmp/exp4.yaml
    ./configs/style_translator_ocahr_cmp/exp5.yaml
    ./configs/style_translator_ocahr_cmp/exp6.yaml
    ./configs/style_translator_ocahr_cmp/exp7.yaml
    ./configs/style_translator_ocahr_cmp/exp8.yaml
)

for config_file in ${config_file_list[@]}; do
for apply_tta in ${apply_tta_list[@]}; do

    CUDA_VISIBLE_DEVICES=3            \
    python evaluate.py                \
        --data_dir    ./data/test     \
        --exp_root    ./experiments   \
        --output_root ./evaluations   \
        --config_file $config_file    \
        --model_name  $model_name     \
        --apply_tta   $apply_tta      \
        --evaluator   $evaluator

done; done
