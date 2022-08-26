# BCI-Challenge

https://bci.grand-challenge.org/

## 1. Environment

```bash
conda create --name bci python=3.8
conda activate bci

# pytorch 1.12.0
# conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

# other packages
pip install -r requirements.txt
```

## 2. Dataset

Download dataset from [here](https://bupt-ai-cz.github.io/BCI_for_GrandChallenge/) and put it in [data](./data) directory.

```
./data
├── test
│   ├── HE
│   ├── IHC
│   └── README.txt
├── train
│   ├── HE
│   ├── IHC
│   └── README.txt
└── val
    ├── HE
    ├── IHC
    └── README.txt
```

## 3. Training

```bash
# basic trainer
CUDA_VISIBLE_DEVICES=0          \
python train.py                 \
    --train_dir   ./data/train  \
    --val_dir     ./data/val    \
    --exp_root    ./experiments \
    --config_file ./configs/style_translator/exp1.yaml  \
    --trainer     basic

# cahr trainer
CUDA_VISIBLE_DEVICES=0          \
python train.py                 \
    --train_dir   ./data/train  \
    --val_dir     ./data/val    \
    --exp_root    ./experiments \
    --config_file ./configs/style_translator_cahr/exp1.yaml \
    --trainer     cahr
```

## 4. Evaluation

```bash
# basic evaluator
CUDA_VISIBLE_DEVICES=0            \
python evaluate.py                \
    --data_dir    ./data/test     \
    --exp_root    ./experiments   \
    --output_root ./evaluations   \
    --config_file ./configs/style_translator/exp1.yaml \
    --model_name  model_best_psnr \
    --apply_tta   true            \
    --evaluator   basic

# cahr evaluator
CUDA_VISIBLE_DEVICES=0            \
python evaluate.py                \
    --data_dir    ./data/test     \
    --exp_root    ./experiments   \
    --output_root ./evaluations   \
    --config_file ./configs/style_translator_cahr/exp1.yaml \
    --model_name  model_best_psnr \
    --apply_tta   true            \
    --evaluator   cahr
```

## 5. Metrics

|      Model       | Exp  | TTA  |  PSNR   |  SSIM  |
| :--------------: | :--: | :--: | :-----: | :----: |
| style_translator | exp4 | true | 22.7937 | 0.5585 |

