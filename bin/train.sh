#! /bin/bash

set -euxo pipefail

python3 src/musicvae/pretrain.py name=fifth
python3 src/musicvae/finetune.py name=eighth pretrain_name=fifth
python3 src/melodyfixer/pretrain.py name=third
python3 src/melodyfixer/finetune.py name=third pretrain_name=third