#! /bin/bash
# These codes are licensed under CC0.
# https://github.com/yskn67/benzaiten/blob/2nd/LICENSE

set -euxo pipefail

python3 src/musicvae/pretrain.py name=fifth
python3 src/musicvae/finetune.py name=twelfth pretrain_name=fifth
python3 src/melodyfixer/pretrain.py name=third
python3 src/melodyfixer/finetune.py name=third pretrain_name=third