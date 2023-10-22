#! /bin/bash

set -euxo pipefail

# python3 src/train.py
python3 src/musicvae/pretrain.py name=first
python3 src/musicvae/finetune.py name=first pretrain_name=first
python3 src/musicvae/pretrain.py name=notranspose train.input_dir="data/preprocess_notranspose"
python3 src/musicvae/finetune.py name=notranspose pretrain_name=notranspose