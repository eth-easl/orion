import torch
import threading
import time
import argparse

from train_info import TrainInfo
from sync_info import SyncInfo
from models.train_imagenet import setup

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training using torchvision models')
parser.add_argument('--policy', default='simple', type=str, help='policy used')

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

        sync_info.barrier.wait()
        start = time.time()

        batch_idx, batch = next(train_iter)

        while batch_idx < 10: #train_size:
            
            # wait for the forward pass of the other thread to finish before starting yours
            if tid == 0:
                sync_info.eventf1.wait()
                sync_info.event_cudaf1.wait(my_stream)
                sync_info.eventf1.clear()
            else:
                sync_info.eventf0.wait()
                sync_info.event_cudaf0.wait(my_stream)
                sync_info.eventf0.clear()

            with torch.cuda.stream(my_stream):
                print(f"time: {time.time()}, thread {tid} starts FORWARD {batch_idx} on stream {my_stream}")
                data, target = batch[0].to(device), batch[1].to(device)
                output = model(data)
                loss = metric_fn(output, target)
                print(f"------------------------ time: {time.time()}, thread {tid} ends FORWARD {batch_idx}")

            # notify that forward is finished
            if tid == 0:
                sync_info.event_cudaf0.record(my_stream)
                sync_info.eventf0.set()
            else:
                sync_info.event_cudaf1.record(my_stream)
                sync_info.eventf1.set()


            # wait for the backward pass of the other thread to finish before starting yours
            if tid == 0:
                sync_info.eventb1.wait()
                sync_info.event_cudab1.wait(my_stream)
                sync_info.eventb1.clear()
            else:
                sync_info.eventb0.wait()
                sync_info.event_cudab0.wait(my_stream)
                sync_info.eventb0.clear()


            with torch.cuda.stream(my_stream):
                print(f"time: {time.time()}, thread {tid} starts BACKWARD {batch_idx} on stream {my_stream}") 
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                print(f"------------------------ time: {time.time()}, thread {tid} ends BACKWARD {batch_idx}")
                batch_idx, batch = next(train_iter)


            # notify that backward pass is finished
            if tid == 0:
                sync_info.event_cudab0.record(my_stream)
                sync_info.eventb0.set()
            else:
                sync_info.event_cudab1.record(my_stream)
                sync_info.eventb1.set()



        barrier.wait()
        end = time.time()
        print(f"TID: {tid}, training took {end-start} sec.")


def train_wrapper_simple(num_epochs, device):
    model, optimizer, train_loader, metric_fn = setup()
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_size = len(train_loader)
        print("train size is: ", train_size)

        train_iter = enumerate(train_loader)

        batch_idx, batch = next(train_iter)

        while batch_idx < train_size:
            optimizer.zero_grad()
            data, target = batch[0].to(device), batch[1].to(device)
            output = model(data)
            loss = metric_fn(output, target)
            loss.backward()
            optimizer.step()
            batch_idx, batch = next(train_iter)


if __name__ == "__main__":

    args = parser.parse_args()
    if args.policy == "tick-tock":
    
        barrier = threading.Barrier(2)

        stream0 = torch.cuda.Stream(device=0)
        stream1 = torch.cuda.Stream(device=0)

        event_cudaf0 = torch.cuda.Event()
        event_cudab0 = torch.cuda.Event()
        eventf0 = threading.Event()
        eventb0 = threading.Event()

        event_cudaf1 = torch.cuda.Event()
        event_cudab1 = torch.cuda.Event()
        eventf1 = threading.Event()
        eventb1 = threading.Event()

        eventf1.set() # t0 starts
        eventb1.set()

        train_dir = '/mnt/data/home/fot/imagenet/imagenet-raw-euwest4/train'
        train_info = TrainInfo('resnet50', 32, 2, 'SGD', train_dir)
        sync_info = SyncInfo(barrier, event_cudaf0, event_cudab0, eventf0, eventb0, event_cudaf1, event_cudab1, eventf1, eventb1)

        thread0 = threading.Thread(target=train_wrapper, args=(stream0, sync_info, 0, 1, 0, train_info))
        thread0.start()

        thread1 = threading.Thread(target=train_wrapper, args=(stream1, sync_info, 1, 1, 0, train_info))
        thread1.start()

        thread0.join()
        thread1.join()

        print("All threads joined!!!!!!!!!!")
