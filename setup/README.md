We have set up a docker image: [fotstrt/orion-ae](https://hub.docker.com/repository/docker/fotstrt/orion-ae/general) with all packages pre-installed.
This directory contains the Dockerfile used to create the image.

If the user does not want to use this image, then please follow these steps:

* Install CUDA 10.2 and CUDNN 7.6.5 (or use a base image containing both, such as: `nvcr.io/nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04` )
* Run `install.sh`
* Install PyTorch from source:
    * `git clone --recursive https://github.com/pytorch/pytorch`
    * `cd pytorch`
    * `git reset --hard 67ece03c8cd632cce9523cd96efde6f2d1cc8121`
    * Apply a patch of changes for Orion: `git apply orion-torch-changes.patch`
    * `git submodule sync`
    * `git submodule update --init --recursive --jobs 0`
    * `python3.8 setup.py develop`

* Install Torchvision from source:
    * `git clone https://github.com/pytorch/vision.git`
    * `cd vision`
    * `git reset --hard da3794e90c7cf69348f5446471926729c55f243e`
    * `python3.8 setup.py develop`
