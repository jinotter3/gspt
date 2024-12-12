# gspt

## GaussianImage representation for CIFAR-10 dataset

![Poster](./assets/poster.jpg)


## Installation
1. Install pytorch

2. Install GaussianImage gsplat package
- check https://github.com/Xinjie-Q/GaussianImage for more details
```bash
git clone https://github.com/XingtongGe/gsplat.git
cd gsplat
pip install -e .[dev]
cd ..
```

3. Install additional packages
```bash
pip install -r requirements.txt
```

## Usage
1. Generate GSCIFAR-10 dataset or download the pre-generated dataset
```bash
cd dataset_generation 
bash run_indiv.sh # modify the script to generate different datasets
```
To download:
https://drive.google.com/drive/folders/1TXP6JhciqT3strimVp4NQEVMhbe1c7Ir?usp=sharing

2. Train the model (point transformer)
```bash
python3 train_pit.py
```