#! /bin/bash

set -euxo pipefail

# python3 src/train.py
# python3 src/musicvae/pretrain.py name=fourth
# python3 src/musicvae/finetune.py name=fifth pretrain_name=fourth
python3 src/melodyfixer/pretrain.py name=second
python3 src/melodyfixer/finetune.py name=second pretrain_name=second