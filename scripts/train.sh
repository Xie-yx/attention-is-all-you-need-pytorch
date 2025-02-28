#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=2,3

# 安装必要的依赖
pip install torch datasets transformers tqdm

# 训练命令
python train.py --batch_size 16 --learning_rate 5e-4 --epochs 12 --output_dir ./wmt14_model
