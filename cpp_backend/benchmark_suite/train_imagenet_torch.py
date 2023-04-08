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

def imagenet_loop(model_name, batchsize, train, num_iters, rps, local_rank, start_barriers, end_barriers, tid):


    if rps > 0:
        sleep_times = np.random.exponential(scale=1/rps, size=num_iters)
    else:
        sleep_times = [0]*num_iters

    s = torch.cuda.Stream()
    timings = []
    start_barriers[tid].wait()

    if True:
        print("-------------- thread id:  ", threading.get_native_id())
        data = torch.rand([batchsize, 3, 224, 224]).to(local_rank)
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

        timings=[]
        with torch.cuda.stream(s):
            for i in range(1):
                print("Start epoch: ", i)

                batch_idx = 0

                while batch_idx < num_iters:

                    print(f"submit!, batch_idx is {batch_idx}")
                    start = time.time()

                    if train:
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                    else:
                        with torch.no_grad():
                            output = model(data)

                    s.synchronize()
                    iter_time = time.time()-start
                    timings.append(iter_time)

                    time.sleep(sleep_times[batch_idx])
                    print(f"{batch_idx} finished, took {iter_time} sec, now sleep for {sleep_times[batch_idx]} sec")

                    batch_idx += 1

                end_barriers[tid].wait()
                # if batch_idx < num_iters-1:
                #     start_barriers[tid].wait()

        timings = timings[2:]
        p50 = np.percentile(timings, 50)
        p95 = np.percentile(timings, 95)
        p99 = np.percentile(timings, 99)

        print(f"Client {tid} finished! p50: {p50} sec, p95: {p95} sec, p99: {p99} sec")
