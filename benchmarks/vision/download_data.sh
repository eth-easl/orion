#!/bin/sh
# a script to download the image net dataset
set -x -e
DATA_DIR=${1:-/cluster/scratch/xianma/vision}
aria2c -c -x 10 -s 10 -d "$DATA_DIR" --download-result=full https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
# now the tar lives in DATA_DIR
cd "$DATA_DIR"
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar
# move the entire compressed file out of the train folder as the folder should contain actual data
mv ILSVRC2012_img_train.tar ../

# the last line should be executed non-interactively (e.g. as a sbatch job) because it is really time-consuming, to unzip each tar
# find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
