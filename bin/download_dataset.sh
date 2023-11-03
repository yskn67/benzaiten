#! /bin/bash
# These codes are licensed under MIT License.
# Copyright (C) 2023 yskn67
# https://github.com/yskn67/benzaiten/blob/2nd/LICENSE

set -euxo pipefail

download_omnibook() {
    DATASET_DIR=$1
    OMNIBOOK_DIR="$DATASET_DIR/omnibook"
    ZIP_PATH="$OMNIBOOK_DIR/omnibook_xml.zip"
    mkdir -p $OMNIBOOK_DIR
    wget -O $ZIP_PATH https://homepages.loria.fr/evincent/omnibook/omnibook_xml.zip
    unzip -od $OMNIBOOK_DIR $ZIP_PATH
    rm -r $OMNIBOOK_DIR/__MACOSX
    rm $ZIP_PATH
}

download_lakh() {
    DATASET_DIR=$1
    LAKH_DIR="$DATASET_DIR/lakh"
    TAR_PATH="$LAKH_DIR/lmd_full.tar.gz"
    mkdir -p $LAKH_DIR
    wget -O $TAR_PATH http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
    tar xvzf $TAR_PATH -C $LAKH_DIR
    rm $TAR_PATH
}

download_lakh_matched() {
    DATASET_DIR=$1
    LAKH_DIR="$DATASET_DIR/lakh_matched"
    TAR_PATH="$LAKH_DIR/lmd_matched.tar.gz"
    mkdir -p $LAKH_DIR
    wget -O $TAR_PATH http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz
    tar xvzf $TAR_PATH -C $LAKH_DIR
    rm $TAR_PATH
}

download_maestro() {
    DATASET_DIR=$1
    MAESTRO_DIR="$DATASET_DIR/maestro"
    ZIP_PATH="$MAESTRO_DIR/maestro-v3.0.0-midi.zip"
    mkdir -p $MAESTRO_DIR
    wget -O $ZIP_PATH https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
    unzip -od $MAESTRO_DIR $ZIP_PATH
    rm $ZIP_PATH
}

download_openewld() {
    DATASET_DIR=$1
    OPENEWLD_DIR="$DATASET_DIR/openewld"
    ZIP_PATH="$OPENEWLD_DIR/v0.1.zip"
    mkdir -p $OPENEWLD_DIR
    wget -O $ZIP_PATH https://github.com/00sapo/OpenEWLD/archive/refs/tags/v0.1.zip
    unzip -od $OPENEWLD_DIR $ZIP_PATH
    rm $ZIP_PATH
}

download_infinite_bach() {
    DATASET_DIR=$1
    BACH_DIR="$DATASET_DIR/infinite_bach"
    mkdir -p $BACH_DIR
    cd $BACH_DIR
    git clone https://github.com/jamesrobertlloyd/infinite-bach
}

download_weimar_midi() {
    DATASET_DIR=$1
    WEIMAR_DIR="$DATASET_DIR/weimar_midi"
    ZIP_PATH="$WEIMAR_DIR/RELEASE2.0_mid_unquant.zip"
    mkdir -p $WEIMAR_DIR
    wget -O $ZIP_PATH https://jazzomat.hfm-weimar.de/download/downloads/RELEASE2.0_mid_unquant.zip
    unzip -od $WEIMAR_DIR $ZIP_PATH
    rm $ZIP_PATH
}


DATASET_DIR="$(cd $(dirname $(dirname $0)); pwd)/data/input"
DATASET_NAME=$1
if [[ "$DATASET_NAME" == "omnibook" ]]; then
    download_omnibook $DATASET_DIR
elif [[ "$DATASET_NAME" == "lakh" ]]; then
    download_lakh $DATASET_DIR
elif [[ "$DATASET_NAME" == "lakh_matched" ]]; then
    download_lakh_matched $DATASET_DIR
elif [[ "$DATASET_NAME" == "maestro" ]]; then
    download_maestro $DATASET_DIR
elif [[ "$DATASET_NAME" == "openewld" ]]; then
    download_openewld $DATASET_DIR
elif [[ "$DATASET_NAME" == "infinite_bach" ]]; then
    download_infinite_bach $DATASET_DIR
elif [[ "$DATASET_NAME" == "weimar_midi" ]]; then
    download_weimar_midi $DATASET_DIR
else
    echo "No action."
fi