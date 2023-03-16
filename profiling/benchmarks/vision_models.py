import os
from platform import node
import sched
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
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

print(torchvision.__file__)

def vision(model_name, batchsize, local_rank, do_eval=True, profile=None):

    data = torch.rand([batchsize, 3, 224, 224]).to(local_rank)
    #data = torch.rand([batchsize, 2048]).to(local_rank)
    model = models.__dict__[model_name](num_classes=1000)
    model = model.to(local_rank)

    print(data.shape)

    if do_eval:
        model.eval()
    else:
        model.train()

    batch_idx = 0
    torch.cuda.synchronize()

    while batch_idx < 10:

        if batch_idx == 9:
            if profile == 'ncu':
                torch.cuda.nvtx.range_push("start")
            elif profile == 'nsys':
                torch.cuda.profiler.cudart().cudaProfilerStart()

        if do_eval:
            with torch.no_grad():
                output = model(data)
        else:
            pass

        if batch_idx == 9:
            if profile == 'ncu':
                torch.cuda.nvtx.range_pop()
            elif profile == 'nsys':
                torch.cuda.profiler.cudart().cudaProfilerStop()

        batch_idx += 1

    print("Done!")

if __name__ == "__main__":
    vision('resnet50', 4, 0, True, 'ncu')
