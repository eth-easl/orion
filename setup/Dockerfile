FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04
WORKDIR /root

RUN apt-get -y update
RUN apt install -y software-properties-common
RUN apt-get install -y vim wget git
RUN apt install -y libjpeg-dev zlib1g-dev tmux build-essential libssl-dev

RUN apt-get -y install python3-pip
RUN python3 -m pip install torch torchvision
RUN python3 -m pip install numpy pandas pyyaml

RUN git clone https://github.com/NVIDIA/DeepLearningExamples.git
COPY nvidia_deeplearning_changes.patch /root/DeepLearningExamples/
RUN cd /root/DeepLearningExamples/PyTorch/LanguageModeling/BERT && export BERT_PREP_WORKING_DIR=/root/DeepLearningExamples/PyTorch/LanguageModeling/BERT && python3 -m pip install -r requirements.txt && python3 -m pip install wget  && bash data/create_datasets_from_start.sh
