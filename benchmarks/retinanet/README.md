un RetinaNet on part of the OpenImages dataset

source: https://github.com/mlcommons/training

1. create a disk from 'image-ssd-openimages' (contains the dataset), attach to the VM and mount
2. git clone https://github.com/mlcommons/training.git
3. cd training/single_stage_detector
4. python3 -m pip install -r requirements.txt
5. cd training/single_stage_detector/ssd
6. python3 train.py --dataset openimages-mlperf --data-path /mnt/openimages/home/fot/openimages_v6/ --batch-size 8
