import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import threading
import argparse

from src.train_info import TrainInfo
from src.sync_info import SyncInfo
import models.train_imagenet as train_imagenet

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training using torchvision models')
parser.add_argument('--policy', default='temporal', type=str, help='policy used')

if __name__ == "__main__":

    args = parser.parse_args()
    train_dir = '/cluster/scratch/xianma/vision/train'
    train_info = TrainInfo(arch='mobilenet_v2', batchsize=32, num_workers=2, optimizer='SGD', train_dir=train_dir)
    device = torch.device("cuda:0")

    stream0 = torch.cuda.Stream(device=device)
    stream1 = torch.cuda.Stream(device=device)

    eventf0 = threading.Event()
    eventb0 = threading.Event()

    eventf1 = threading.Event()
    eventb1 = threading.Event()

    eventf1.set()  # t0 starts
    eventb1.set()

    sync_info = SyncInfo(eventf0, eventb0, eventf1, eventb1)

    if args.policy == "tick-tock":
        thread0 = threading.Thread(target=train_imagenet.train_wrapper, kwargs={
            'my_stream': stream0,
            'sync_info': sync_info,
            'tid': 0,
            'num_epochs': 5,
            'device': device,
            'train_info': train_info
        })
        thread0.start()
        thread1 = threading.Thread(target=train_imagenet.train_wrapper, kwargs={
            'my_stream': stream1,
            'sync_info': sync_info,
            'tid': 1,
            'num_epochs': 5,
            'device': device,
            'train_info': train_info
        })
        thread1.start()

        thread0.join()
        thread1.join()
        print("All threads joined!!!!!!!!!!")

    elif args.policy == "temporal":
        sync_info.no_sync_control = True
        train_imagenet.train_wrapper(my_stream=stream0, sync_info=sync_info, tid=0, num_epochs=5, device=device,
                                     train_info=train_info)
        train_imagenet.train_wrapper(my_stream=stream1, sync_info=sync_info, tid=1, num_epochs=5, device=device,
                                     train_info=train_info)
