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


def imagenet_loop(model_name, batchsize, train, local_rank, barriers, tid):

    print(model_name, batchsize, local_rank, barriers, tid)

    # do only forward for now, experimental
    barriers[0].wait()
    ds = torch.cuda.default_stream()

    #barriers[0].wait()

    print("-------------- thread id:  ", threading.get_native_id())

    data = torch.rand([batchsize, 3, 224, 224]).to(local_rank).contiguous()
    #data = torch.rand([batchsize, 2048]).to(local_rank)
    model = models.__dict__[model_name](num_classes=1000)
    model = model.to(0)

    if train:
        model.train()
    else:
        model.eval()

    print("Enter loop!")

    s = torch.cuda.Stream()
    #with torch.cuda.stream(s):
    if True:
        timings=[]
        for i in range(10):
            print("Start epoch: ", i)

            start = time.time()
            start_iter = time.time()

            batch_idx = 0

            while batch_idx < 1:

                print(f"submit!, batch_idx is {batch_idx}")
                #torch.cuda.profiler.cudart().cudaProfilerStart()

                if train:
                    output = model(data)
                else:
                    with torch.no_grad():
                        output = model(data)

                #torch.cuda.profiler.cudart().cudaProfilerStop()

                batch_idx += 1

                start_iter = time.time()

                #torch.cuda.synchronize()
                # wait until all operations have been completed before starting the next iter


            #barriers[0].wait()
            if i < 9:
                barriers[0].wait()
                #barriers[0].wait()
            print(f"{tid}, Epoch done!")

        print("Finished! Ready to join!")