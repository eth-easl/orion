import torch
import threading
import time
import argparse
import logging
from sync_controller import *

from train_info import TrainInfo
from sync_info import SyncInfo
from models.train_imagenet import setup
from datetime import datetime
logging.basicConfig(filename='/cluster/scratch/xianma/log/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log', level=logging.DEBUG)
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training using torchvision models')
parser.add_argument('--policy', default='temporal', type=str, help='policy used')

def train_wrapper(my_stream, sync_info, tid, num_epochs, device, train_info):
    print(f"thread {tid} starts!!")

    model, optimizer, train_loader, metric_fn = setup(train_info, device)
    model = model.to(device)
    model.train()
    print(f"training {tid} starts!!")

    for epoch in range(num_epochs):

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
    logging.info('enter simple wrapper')
    print('enter simple wrapper')
    model, optimizer, train_loader, metric_fn = setup(train_info, device)
    model = model.to(device)
    model.train()
    logging.info('model is moved to gpu')
    loss_sum = 0
    for epoch in range(num_epochs):
        train_size = len(train_loader)
        logging.info(f'train size is {train_size}')

        start = time.time()
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            data, target = batch[0].to(device), batch[1].to(device)
            output = model(data)
            loss = metric_fn(output, target)
            loss_sum += loss.item()
            if batch_idx % 100 == 0:
                logging.info(f"loss at {batch_idx} iteration is {loss_sum / 100}")
                loss_sum = 0
            loss.backward()
            optimizer.step()

        end = time.time()
        logging.info(f"Training took: {end - start} sec.")


if __name__ == "__main__":

    args = parser.parse_args()
    train_dir = '/cluster/scratch/xianma/vision/train'
    train_info = TrainInfo(arch='mobilenet_v2', batchsize=32, num_workers=2, optimizer='SGD', train_dir=train_dir)
    device = torch.device("cuda:0")
    if args.policy == "tick-tock":

        stream0 = torch.cuda.Stream(device=device)
        stream1 = torch.cuda.Stream(device=device)

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

        train_wrapper_simple(train_info, num_epochs=3, device=device)
        # train_wrapper_simple(train_info, num_epochs=5, device=device)
