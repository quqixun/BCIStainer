# BCI-Challenge

Challenge page: https://bci.grand-challenge.org/

## 1. Environment

```bash
conda create --name bci python=3.8
conda activate bci

# pytorch 1.12.0
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

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

## 3. Training and Evaluation

```bash
# training
CUDA_VISIBLE_DEVICES=0          \
python train.py                 \
    --train_dir   ./data/train  \
    --val_dir     ./data/val    \
    --exp_root    ./experiments \
    --config_file $config_file  \
    --trainer     basic  # basic or cahr according to configs

# evaluation
CUDA_VISIBLE_DEVICES=0            \
python evaluate.py                \
    --data_dir    ./data/test     \
    --exp_root    ./experiments   \
    --output_root ./evaluations   \
    --config_file $config_file    \
    --model_name  model_best_psnr \
    --apply_tta   true            \
    --evaluator   basic  # basic or cahr according to configs
```

## 5. Metrics

<table>
    <thead>
        <tr>
            <th>No.</th>
            <th>Configs</th>
            <th>Stainer</th>
            <th>Style</th>
            <th>Attention</th>
            <th>Comparator</th>
            <th>TTA</th>
            <th>PSNR</th>
            <th>SSIM</th>
            <th>Comment</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="2">1</td>
            <td rowspan="2"><a href="./configs/stainer_basic/exp1.yaml">stainer_basic/exp1</a></td>
            <td rowspan="2">basic</td>
            <td rowspan="2">mod</td>
            <td rowspan="2">x</td>
            <td rowspan="2">x</td>
            <td>x</td>
            <td>22.2289</td>
            <td>0.5294</td>
            <td>&nbsp;</td>
        </tr>
        <tr>
            <td>o</td>
            <td>22.6266</td>
            <td>0.5737</td>
            <td>&nbsp;</td>
        </tr>
        <tr>
            <td rowspan="2">2</td>
            <td rowspan="2"><a href="./configs/stainer_basic/exp2.yaml">stainer_basic/exp2</a></td>
            <td rowspan="2">basic</td>
            <td rowspan="2">adain</td>
            <td rowspan="2">x</td>
            <td rowspan="2">x</td>
            <td>x</td>
            <td>22.7732</td>
            <td>0.5245</td>
            <td>&nbsp;</td>
        </tr>
        <tr>
            <td>o</td>
            <td>23.2413</td>
            <td>0.5726</td>
            <td>&nbsp;</td>
        </tr>
        <tr>
            <td rowspan="2">3</td>
            <td rowspan="2"><a href="./configs/stainer_basic/exp3.yaml">stainer_basic/exp3</a></td>
            <td rowspan="2">basic</td>
            <td rowspan="2">mod</td>
            <td rowspan="2">o</td>
            <td rowspan="2">x</td>
            <td>x</td>
            <td>22.5492</td>
            <td>0.5312</td>
            <td>&nbsp;</td>
        </tr>
        <tr>
            <td>o</td>
            <td>22.8406</td>
            <td>0.5646</td>
            <td>&nbsp;</td>
        </tr>
        <tr>
            <td rowspan="2">4</td>
            <td rowspan="2"><a href="./configs/stainer_basic/exp4.yaml">stainer_basic/exp4</a></td>
            <td rowspan="2">basic</td>
            <td rowspan="2">adain</td>
            <td rowspan="2">o</td>
            <td rowspan="2">x</td>
            <td>x</td>
            <td>22.5447</td>
            <td>0.5316</td>
            <td>&nbsp;</td>
        </tr>
        <tr>
            <td>o</td>
            <td>22.9690</td>
            <td>0.5760</td>
            <td>&nbsp;</td>
        </tr>
        <tr>
            <td rowspan="2">5</td>
            <td rowspan="2"><a href="./configs/stainer_basic_cmp/exp1.yaml">stainer_basic_cmp/exp1</a></td>
            <td rowspan="2">basic</td>
            <td rowspan="2">mod</td>
            <td rowspan="2">x</td>
            <td rowspan="2">basic</td>
            <td>x</td>
            <td>22.3711</td>
            <td>0.5293</td>
            <td>&nbsp;</td>
        </tr>
        <tr>
            <td>o</td>
            <td>22.7570</td>
            <td>0.5743</td>
            <td>&nbsp;</td>
        </tr>
        <tr>
            <td rowspan="2">6</td>
            <td rowspan="2"><a href="./configs/stainer_basic_cmp/exp2.yaml">stainer_basic_cmp/exp2</a></td>
            <td rowspan="2">basic</td>
            <td rowspan="2">adain</td>
            <td rowspan="2">x</td>
            <td rowspan="2">basic</td>
            <td>x</td>
            <td>22.8123</td>
            <td>0.5273</td>
            <td>&nbsp;</td>
        </tr>
        <tr>
            <td>o</td>
            <td>23.3942</td>
            <td>0.5833</td>
            <td>best in metrics,<br>droplet artifacts and blur</td>
        </tr>
        <tr>
            <td rowspan="2">7</td>
            <td rowspan="2"><a href="./configs/stainer_basic_cmp/exp3.yaml">stainer_basic_cmp/exp3</a></td>
            <td rowspan="2">basic</td>
            <td rowspan="2">mod</td>
            <td rowspan="2">o</td>
            <td rowspan="2">basic</td>
            <td>x</td>
            <td>22.5357</td>
            <td>0.5175</td>
            <td>&nbsp;</td>
        </tr>
        <tr>
            <td>o</td>
            <td>22.9293</td>
            <td>0.5585</td>
            <td>best in visual,<br>shadow artifacts</td>
        </tr>
        <tr>
            <td rowspan="2">8</td>
            <td rowspan="2"><a href="./configs/stainer_basic_cmp/exp4.yaml">stainer_basic_cmp/exp4</a></td>
            <td rowspan="2">basic</td>
            <td rowspan="2">adain</td>
            <td rowspan="2">o</td>
            <td rowspan="2">basic</td>
            <td>x</td>
            <td>22.5447</td>
            <td>0.5316</td>
            <td>&nbsp;</td>
        </tr>
        <tr>
            <td>o</td>
            <td>22.9809</td>
            <td>0.5697</td>
            <td>in metrics: 6 < 8 < 7,<br>in visula: 7 < 8 < 6 </td>
        </tr>
    </tbody>
</table>





