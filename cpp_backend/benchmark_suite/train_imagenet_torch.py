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

def imagenet_loop(model_name, batchsize, train, local_rank, start_barriers, end_barriers, tid):

    # do only forward for now, experimental
    s = torch.cuda.Stream()
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
        #if True:
            for i in range(10):
                print("Start epoch: ", i)

                start = time.time()
                start_iter = time.time()

                batch_idx = 0

                while batch_idx < 1:

                    print(f"submit!, batch_idx is {batch_idx}")
                    torch.cuda.profiler.cudart().cudaProfilerStart()

                    if train:
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                    else:
                        with torch.no_grad():
                            output = model(data)

                    torch.cuda.profiler.cudart().cudaProfilerStop()

                    batch_idx += 1

                    start_iter = time.time()

                print("Epoch done!")
                end_barriers[tid].wait()
                if i < 9:
                    start_barriers[tid].wait()

        # timings = timings[2:]
        # print(f"Avg is {np.median(np.asarray(timings))} sec")
        print("Finished! Ready to join!")

#imagenet_loop('resnet50', 32, True, 0, None, 0)