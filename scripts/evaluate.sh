#!/bin/bash


# settings
device=0
apply_tta=true
evaluator=basic
model_name=model_best_psnr

# configurations of experiment
config_file=./configs/stainer_basic_cmp/exp3.yaml

# evaluation
CUDA_VISIBLE_DEVICES=$device      \
python evaluate.py                \
    --data_dir    ./data/test     \
    --exp_root    ./experiments   \
    --output_root ./evaluations   \
    --config_file $config_file    \
    --model_name  $model_name     \
    --apply_tta   $apply_tta      \
    --evaluator   $evaluator
