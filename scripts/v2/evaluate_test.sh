#!/bin/bash


config_file_list=(
    # ./configs/resnet_ada_v2/exp4.yaml
    # ./configs/resnet_ada_v2/exp5.yaml
    # ./configs/resnet_ada_v2/exp6.yaml
    # ./configs/resnet_ada_v2/exp7.yaml
    # ./configs/resnet_ada_v2/exp8.yaml
    # ./configs/resnet_ada_v2/exp9.yaml
    # ./configs/resnet_ada_v2/exp10.yaml
    # ./configs/resnet_ada_v2/exp11.yaml
    # ./configs/resnet_ada_v2/exp12.yaml
    # ./configs/resnet_ada_v2/exp13.yaml
    # ./configs/resnet_ada_v2/exp14.yaml
    # ./configs/resnet_ada_v2/exp15.yaml
    # ./configs/resnet_ada_v2/exp16.yaml
    # ./configs/resnet_ada_v2/exp17.yaml
    # ./configs/resnet_ada_v2/exp18.yaml
    # ./configs/resnet_ada_v2/exp19.yaml
    # ./configs/resnet_ada_v2/exp20.yaml
    # ./configs/resnet_ada_l_v2/exp1.yaml
    # ./configs/resnet_ada_l_v2/exp2.yaml
    # ./configs/resnet_ada_l_v2/exp3.yaml
    # ./configs/resnet_ada_l_v2/exp4.yaml
    # ./configs/resnet_ada_l_v2/exp5.yaml
    # ./configs/resnet_ada_h_v2/exp1.yaml
    # ./configs/resnet_ada_h_v2/exp2.yaml
    # ./configs/resnet_ada_h_v2/exp3.yaml
    # ./configs/resnet_ada_h_v2/exp4.yaml
    # ./configs/resnet_ada_h_v2/exp5.yaml
    # ./configs/resnet_mod_v2/exp1.yaml
    # ./configs/resnet_mod_v2/exp2.yaml
    # ./configs/resnet_mod_v2/exp3.yaml
    ./configs/resnet_mod_v2/exp4.yaml
    ./configs/resnet_mod_v2/exp5.yaml
    ./configs/resnet_mod_v2/exp6.yaml
)


for config_file in ${config_file_list[@]}; do

    CUDA_VISIBLE_DEVICES=0            \
    python evaluate.py                \
        --data_dir    ./data/test     \
        --exp_root    ./experiments   \
        --output_root ./outputs_test  \
        --config_file $config_file    \
        --model_name  model_best_psnr \
        --apply_tta

done
