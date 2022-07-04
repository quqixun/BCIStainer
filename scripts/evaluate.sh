#!/bin/sh


# config_list = [
#     ./confgis/baseline.ymal
#     ...
# ]


CUDA_VISIBLE_DEVICES=0                    \
python evaluate.py                        \
    --data_dir    ./data/val              \
    --exp_root    ./experiments           \
    --output_root ./outputs               \
    --config_file ./configs/baseline.yaml
