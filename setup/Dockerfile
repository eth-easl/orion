FROM nvcr.io/nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
WORKDIR /root

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get -y update
RUN apt install -y software-properties-common
RUN apt-get install -y vim wget git
RUN apt install -y libjpeg-dev zlib1g-dev


RUN apt -y install build-essential libssl-dev
RUN wget https://github.com/Kitware/CMake/releases/download/v3.19.6/cmake-3.19.6.tar.gz
RUN tar -zxvf cmake-3.19.6.tar.gz
RUN cd cmake-3.19.6 && ./bootstrap && make && make install && cp bin/cmake /bin/  && cd ..

RUN apt update -y
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.8-dev -y

RUN apt-get -y install python3-pip
RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install pyyaml typing_extensions
RUN python3.8 -m pip install Pillow
RUN python3.8 -m pip install numpy

RUN git clone --recursive https://github.com/pytorch/pytorch
COPY orion-torch-changes.patch /root/pytorch/
RUN cd pytorch && git reset --hard 67ece03c8cd632cce9523cd96efde6f2d1cc8121 && git apply orion-torch-changes.patch && git submodule sync && git submodule update --init --recursive --jobs 0 && python3.8 setup.py develop && cd ..

RUN git clone https://github.com/pytorch/vision.git
RUN cd vision && git reset --hard da3794e90c7cf69348f5446471926729c55f243e && python3.8 setup.py develop && cd ..

RUN echo "alias python=python3.8" >> /root/.bashrc
SHELL ["source" , "/root/.bashrc"]
SHELL ["/bin/sh", "-c"]


RUN git clone https://github.com/NVIDIA/DeepLearningExamples.git
COPY nvidia_deeplearning_changes.patch /root/DeepLearningExamples/

RUN cd DeepLearningExamples/ && git reset --hard 6610c05c330b887744993fca30532cbb9561cbde && git apply nvidia_deeplearning_changes.patch
RUN cd /root/DeepLearningExamples/PyTorch/LanguageModeling/BERT && export BERT_PREP_WORKING_DIR=/root/DeepLearningExamples/PyTorch/LanguageModeling/BERT && python3.8 -m pip install -r requirements.txt && python3.8 -m pip install wget  && bash data/create_datasets_from_start.sh

RUN cd /root/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL && pip install -r requirements.txt &&  bash getdata.sh
