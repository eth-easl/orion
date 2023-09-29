#!/bin/bash

sudo apt-get update
sudo apt install software-properties-common
sudo apt-get install vim wget git
sudo apt install libjpeg-dev zlib1g-dev

# cmake

sudo apt install build-essential libssl-dev
wget https://github.com/Kitware/CMake/releases/download/v3.19.6/cmake-3.19.6.tar.gz
tar -zxvf cmake-3.19.6.tar.gz
cd cmake-3.19.6
./bootstrap
make
sudo make install
cp bin/cmake /bin/
cd ..

# python

sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.8-dev

# pip

sudo apt-get -y install python3-pip
python3.8 -m pip install --upgrade pip
python3.8 -m pip install pyyaml typing_extensions
python3.8 -m pip install Pillow
python3.8 -m pip install numpy