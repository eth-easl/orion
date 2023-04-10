import os
from platform import node
import sched
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import models, datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock
from datetime import timedelta
import random
import numpy as np
import time
import os
import argparse
import threading

class DummyDataLoader():
    def __init__(self, batchsize):
        self.batchsize = batchsize
        self.data = torch.rand([self.batchsize, 3, 224, 224]).contiguous()
        self.target = torch.ones([self.batchsize]).to(torch.long)

    def __iter__(self):
        return self

    def __next__(self):
        return self.data, self.target

class RealDataLoader():
    def __init__(self, batchsize):
        train_transform =  transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))]
        )
        train_dataset = \
                datasets.ImageFolder("/mnt/data/home/fot/imagenet/imagenet-raw-euwest4",transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batchsize, num_workers=8)

    def __iter__(self):
        print("Inside iter")
        return iter(self.train_loader)



def imagenet_loop(model_name, batchsize, train, num_iters, rps, dummy_data, local_rank, barriers, tid):

    print(model_name, batchsize, local_rank, barriers, tid)

    # do only forward for now, experimental
    barriers[0].wait()

    #if tid==1:
    #    time.sleep(5)
    if rps > 0:
        sleep_times = np.random.exponential(scale=1/rps, size=num_iters)
    else:
        sleep_times = [0]*num_iters

    print("-------------- thread id:  ", threading.get_native_id())

    #data = torch.rand([batchsize, 3, 224, 224]).contiguous()
    #target = torch.ones([batchsize]).to(torch.long)
    model = models.__dict__[model_name](num_classes=1000)
    model = model.to(0)

    if train:
        model.train()
        optimizer =  torch.optim.SGD(model.parameters(), lr=0.1)
        criterion =  torch.nn.CrossEntropyLoss().to(local_rank)
    else:
        model.eval()

    if dummy_data:
        train_loader = DummyDataLoader(batchsize)
    else:
        train_loader = RealDataLoader(batchsize)

    train_iter = enumerate(train_loader)
    batch_idx, batch = next(train_iter)

    gpu_data, gpu_target = batch[0].to(local_rank), batch[1].to(local_rank)
    print("Enter loop!")


    if True:
        timings=[]
        for i in range(1):
            print("Start epoch: ", i)

            start = time.time()
            start_iter = time.time()

            while batch_idx < num_iters:

                print(f"submit!, batch_idx is {batch_idx}")
                #torch.cuda.profiler.cudart().cudaProfilerStart()

                if train:
                    gpu_data, gpu_target = batch[0].to(local_rank), batch[1].to(local_rank)
                    optimizer.zero_grad()
                    output = model(gpu_data)
                    loss = criterion(output, gpu_target)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        gpu_data = batch[0].to(local_rank)
                        output = model(gpu_data)
                #torch.cuda.profiler.cudart().cudaProfilerStop()

                time.sleep(sleep_times[batch_idx])
                #print(f"{batch_idx} submitted! sent everything, sleep for {sleep_times[batch_idx]} sec")

                batch_idx, batch = next(train_iter)
                if (batch_idx == 1): # for backward
                    barriers[0].wait()


                # if batch_idx < num_iters:
                #     barriers[0].wait()


        print("Finished! Ready to join!")
