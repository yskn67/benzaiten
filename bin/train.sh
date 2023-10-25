#! /bin/bash

set -euxo pipefail

# python3 src/train.py
python3 src/musicvae/pretrain.py name=second
python3 src/musicvae/pretrain.py name=second_nt train.input_dir="data/preprocess_notranspose"
python3 src/musicvae/finetune.py name=second pretrain_name=second
python3 src/musicvae/finetune.py name=second_nt pretrain_name=second_nt