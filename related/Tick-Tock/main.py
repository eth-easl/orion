import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import threading
import time
import argparse
from sync_controller import *
from datetime import datetime
from train_info import TrainInfo
from sync_info import SyncInfo
from models.train_imagenet import setup

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training using torchvision models')
parser.add_argument('--policy', default='temporal', type=str, help='policy used')


def pretty_time():
    return datetime.now().strftime('%H:%M:%S')


def train_wrapper(my_stream, sync_info, tid, num_epochs, device, train_info):
    model, optimizer, train_loader, metric_fn = setup(train_info, device)
    model.train()
    print(f"training {tid} starts!!")

    start = time.time()
    loss_sum = 0
    print_every = 50
    for epoch in range(num_epochs):

        train_size = len(train_loader)
        print("train size is: ", train_size)

        for batch_idx, batch in enumerate(train_loader):
            with ForwardController(thread_id=tid, sync_info=sync_info):
                print(f"time: {pretty_time()}, thread {tid} starts FORWARD {batch_idx}")
                with torch.cuda.stream(my_stream):
                    data, target = batch[0].to(device), batch[1].to(device)
                    output = model(data)
                    loss = metric_fn(output, target)
                    loss_sum += loss.item()
                print(f"time: {pretty_time()}, thread {tid} ends FORWARD {batch_idx}")

            if batch_idx % print_every == 0:
                print(f"loss for thread {tid}: {loss_sum / print_every}")
                loss_sum = 0

            with BackwardController(thread_id=tid, sync_info=sync_info):
                print(f"time: {pretty_time()}, thread {tid} starts BACKWARD {batch_idx}")
                with torch.cuda.stream(my_stream):
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                print(f"time: {pretty_time()}, thread {tid} ends BACKWARD {batch_idx}")
    sync_info.no_sync_control = True
    end = time.time()
    print(f"TID: {tid}, training took {end - start} sec.")


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
        thread0 = threading.Thread(target=train_wrapper, kwargs={
            'my_stream': stream0,
            'sync_info': sync_info,
            'tid': 0,
            'num_epochs': 5,
            'device': device,
            'train_info': train_info
        })
        thread0.start()
        thread1 = threading.Thread(target=train_wrapper, kwargs={
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
        train_wrapper(my_stream=stream0, sync_info=sync_info, tid=0, num_epochs=5, device=device, train_info=train_info)
        train_wrapper(my_stream=stream1, sync_info=sync_info, tid=1, num_epochs=5, device=device, train_info=train_info)
