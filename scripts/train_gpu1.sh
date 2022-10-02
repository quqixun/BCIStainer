#!/bin/bash


# settings
device=1
trainer=basic  # basic, cahr
evaluator=$trainer
model_name=model_best_psnr
apply_tta_list=(false true)

# configurations of experiments
config_file_list=(
    # ./configs/stainer_basic/exp3.yaml       # basic
    # ./configs/stainer_basic/exp4.yaml       # basic
    # ./configs/stainer_cahr/exp2.yaml        # cahr
    ./configs/stainer_basic_cmp/exp6.yaml   # basic
    ./configs/stainer_basic_cmp/exp10.yaml  # basic
)

for config_file in ${config_file_list[@]}; do

    # training
    CUDA_VISIBLE_DEVICES=$device    \
    python train.py                 \
        --train_dir   ./data/train  \
        --val_dir     ./data/val    \
        --exp_root    ./experiments \
        --config_file $config_file  \
        --trainer     $trainer

    # evaluation
    for apply_tta in ${apply_tta_list[@]}; do

        CUDA_VISIBLE_DEVICES=$device      \
        python evaluate.py                \
            --data_dir    ./data/test     \
            --exp_root    ./experiments   \
            --output_root ./evaluations   \
            --config_file $config_file    \
            --model_name  $model_name     \
            --apply_tta   $apply_tta      \
            --evaluator   $evaluator

    done

done
