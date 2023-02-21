import torch
import threading
import time
import argparse
from sync_controller import *

from train_info import TrainInfo
from sync_info import SyncInfo
from models.train_imagenet import setup

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training using torchvision models')
parser.add_argument('--policy', default='temporal', type=str, help='policy used')


def train_wrapper(my_stream, sync_info, tid, num_epochs, device, train_info):
    print(f"thread {tid} starts!!")

    model, optimizer, train_loader, metric_fn = setup(train_info, device)
    model = model.to(device)

    print(f"training {tid} starts!!")

    for epoch in range(num_epochs):
        model.train()
        train_size = len(train_loader)
        print("train size is: ", train_size)

        train_iter = enumerate(train_loader)
        start = time.time()

        batch_idx, batch = next(train_iter)

        while batch_idx < 1000:  # train_size:

            # print(f"Thread {tid} started iteration {batch_idx}")

            with ForwardController(thread_id=tid, sync_info=sync_info):
                with torch.cuda.stream(my_stream):
                    # print(f"time: {time.time()}, thread {tid} starts FORWARD {batch_idx} on stream {my_stream}")
                    data, target = batch[0].to(device), batch[1].to(device)
                    output = model(data)
                    loss = metric_fn(output, target)
                    # print(f"------------------------ time: {time.time()}, thread {tid} ends FORWARD {batch_idx}")

            with BackwardController(thread_id=tid, sync_info=sync_info):
                with torch.cuda.stream(my_stream):
                    # print(f"time: {time.time()}, thread {tid} starts BACKWARD {batch_idx} on stream {my_stream}")
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # print(f"------------------------ time: {time.time()}, thread {tid} ends BACKWARD {batch_idx}")
                    batch_idx, batch = next(train_iter)

        end = time.time()
        print(f"TID: {tid}, training took {end - start} sec.")


def train_wrapper_simple(train_info, num_epochs, device):
    model, optimizer, train_loader, metric_fn = setup(train_info, device)
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_size = len(train_loader)
        print("train size is: ", train_size)

        train_iter = enumerate(train_loader)

        start = time.time()

        batch_idx, batch = next(train_iter)

        while batch_idx < 1000:  # train_size:
            # print(f"entered iteration {batch_idx}")
            optimizer.zero_grad()
            data, target = batch[0].to(device), batch[1].to(device)
            output = model(data)
            loss = metric_fn(output, target)
            loss.backward()
            optimizer.step()
            batch_idx, batch = next(train_iter)
        end = time.time()
        print(f"Training took: {end - start} sec.")


if __name__ == "__main__":

    args = parser.parse_args()

    train_dir = '/mnt/data/home/fot/imagenet/imagenet-raw-euwest4/train'
    train_info = TrainInfo(arch='mobilenet_v2', batchsize=32, num_workers=8, optimizer='SGD', train_dir=train_dir)

    if args.policy == "tick-tock":

        stream0 = torch.cuda.Stream(device=0)
        stream1 = torch.cuda.Stream(device=0)

        eventf0 = threading.Event()
        eventb0 = threading.Event()

        eventf1 = threading.Event()
        eventb1 = threading.Event()

        eventf1.set()  # t0 starts
        eventb1.set()

        sync_info = SyncInfo(eventf0, eventb0, eventf1, eventb1)

        thread0 = threading.Thread(target=train_wrapper, args=(stream0, sync_info, 0, 5, 0, train_info))
        thread0.start()

        thread1 = threading.Thread(target=train_wrapper, args=(stream1, sync_info, 1, 5, 0, train_info))
        thread1.start()

        thread0.join()
        thread1.join()

        print("All threads joined!!!!!!!!!!")

    elif args.policy == "temporal":

        train_wrapper_simple(train_info, num_epochs=5, device=0)
        train_wrapper_simple(train_info, num_epochs=5, device=0)
