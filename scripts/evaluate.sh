#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=2

# 安装必要的依赖
pip install torch datasets transformers tqdm nltk

# 评估命令
python evaluate.py --batch_size 16 --model_dir ./wmt14_model --device cuda
