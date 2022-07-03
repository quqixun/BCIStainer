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

