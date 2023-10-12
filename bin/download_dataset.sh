#! /bin/bash

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

DATASET_DIR="$(cd $(dirname $(dirname $0)); pwd)/data/input"
DATASET_NAME=$1
if [[ "$DATASET_NAME" == "omnibook" ]]; then
    download_omnibook $DATASET_DIR
elif [[ "$DATASET_NAME" == "lakh" ]]; then
    download_lakh $DATASET_DIR
else
    echo "No action."
fi