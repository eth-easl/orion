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


def imagenet_loop(model_name, batchsize, train, num_iters, rps, local_rank, barriers, tid):

    print(model_name, batchsize, local_rank, barriers, tid)

    # do only forward for now, experimental
    barriers[0].wait()

    #if tid==1:
    #    time.sleep(5)
    if rps > 0:
        sleep_times = np.random.exponential(scale=1/rps, size=num_iters)
    else:
        sleep_times = [0]*num_iters

    ds = torch.cuda.default_stream()

    #barriers[0].wait()

    print("-------------- thread id:  ", threading.get_native_id())

    data = torch.rand([batchsize, 3, 224, 224]).to(local_rank).contiguous()
    target = torch.ones([batchsize]).to(torch.long).to(local_rank)
    #data = torch.rand([batchsize, 2048]).to(local_rank)
    model = models.__dict__[model_name](num_classes=1000)
    model = model.to(0)

    if train:
        model.train()
        optimizer =  torch.optim.SGD(model.parameters(), lr=0.1)
        criterion =  torch.nn.CrossEntropyLoss().to(local_rank)
    else:
        model.eval()


    print("Enter loop!")

    s = torch.cuda.Stream()

    #with torch.cuda.stream(s):
    if True:
        timings=[]
        for i in range(1):
            print("Start epoch: ", i)

            start = time.time()
            start_iter = time.time()

            batch_idx = 0

            while batch_idx < num_iters:

                print(f"submit!, batch_idx is {batch_idx}")
                #torch.cuda.profiler.cudart().cudaProfilerStart()

                if train:
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        output = model(data)
                #torch.cuda.profiler.cudart().cudaProfilerStop()

                time.sleep(sleep_times[batch_idx])
                print(f"{batch_idx} submitted! sent everything, sleep for {sleep_times[batch_idx]} sec")

                batch_idx += 1

                if (batch_idx == 1): # for backward
                    barriers[0].wait()


                # if batch_idx < num_iters:
                #     barriers[0].wait()


        print("Finished! Ready to join!")