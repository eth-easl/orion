from train_info import TrainInfo
import torch
import threading
import time
import argparse

from models.train_imagenet import setup

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training using torchvision models')
parser.add_argument('--policy', default='simple', type=str, help='policy used')

def train_wrapper(barrier, my_stream, event_cudaf0, event0, event_cudaf1, event1, tid, num_epochs, device, train_info):

    print(f"thread {tid} starts!!")

    model, optimizer, train_loader, metric_fn = setup(train_info, device)
    model = model.to(device)

    print(f"training {tid} starts!!")

    for epoch in range(num_epochs):
        model.train()
        train_size = len(train_loader)
        print("train size is: ", train_size)

        train_iter = enumerate(train_loader)

        barrier.wait()
        start = time.time()

        batch_idx, batch = next(train_iter)

        while batch_idx < 100: #train_size:
            
            if tid == 0:
                event1.wait()
                event_cudaf1.wait(my_stream)
                event1.clear()
            else:
                event0.wait()
                event_cudaf0.wait(my_stream)
                event0.clear()

            with torch.cuda.stream(my_stream):
                print(f"thread {tid} starts FORWARD {batch_idx} on stream {my_stream}")
                data, target = batch[0].to(device), batch[1].to(device)
                output = model(data)
                loss = metric_fn(output, target)

            if tid == 0:
                event_cudaf0.record(my_stream)
                event0.set()
            else:
                event_cudaf1.record(my_stream)
                event1.set()

            with torch.cuda.stream(my_stream):
                print(f"thread {tid} starts BACKWARD {batch_idx} on stream {my_stream}") 
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_idx, batch = next(train_iter)

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
        event0 = threading.Event()

        event_cudaf1 = torch.cuda.Event()
        event1 = threading.Event()
        event1.set() # thread0 starts

        train_dir = '/mnt/data/home/fot/imagenet/imagenet-raw-euwest4/train'
        train_info = TrainInfo('resnet50', 32, 2, 'SGD', train_dir)

        thread0 = threading.Thread(target=train_wrapper, args=(barrier, stream0, event_cudaf0, event0, event_cudaf1, event1, 0, 1, 0, train_info))
        thread0.start()

        thread1 = threading.Thread(target=train_wrapper, args=(barrier, stream1, event_cudaf0, event0, event_cudaf1, event1, 1, 1, 0, train_info))
        thread1.start()

        thread0.join()
        thread1.join()

        print("All threads joined!!!!!!!!!!")
