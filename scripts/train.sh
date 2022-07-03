#!/bin/sh


# config_list = [
#     ./confgis/baseline.ymal
#     ...
# ]


CUDA_VISIBLE_DEVICES=3                    \
python train.py                           \
    --train_dir   ./data/train            \
    --val_dir     ./data/val              \
    --exp_root    ./experiments           \
    --config_file ./configs/baseline.yaml
