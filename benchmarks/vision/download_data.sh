#!/bin/sh
# a script to download the image net dataset
cd /cluster/scratch/xianma
rm -rf vision
aria2c -c -x 10 -s 10 -d vision --download-result=full https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
# now the tar lives in ./vision
cd vision
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar
mv ILSVRC2012_img_train.tar ../

# the last line should be executed non-interactively because it is really time-consuming, to unzip each tar
# find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
