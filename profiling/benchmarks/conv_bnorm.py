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

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3,64,kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = torch.nn.BatchNorm2d(64)

    def forward(self, x):
        for i in range(25):
            y = self.conv(x)
            z = self.bn(y)


def conv_bnorm_loop(batchsize, local_rank, do_eval=True, profile=None):

    print("-------------- thread id:  ", threading.get_native_id())

    data = torch.rand([batchsize, 3, 224, 224]).to(local_rank).contiguous()
    model = Model()
    model = model.to(0)

    if do_eval:
        model.eval()
    else:
        model.train()

    print("Enter loop!")

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
            output = model(data)

        if batch_idx == 9:
            if profile == 'ncu':
                torch.cuda.nvtx.range_pop()
            elif profile == 'nsys':
                torch.cuda.profiler.cudart().cudaProfilerStop()

        batch_idx += 1

    print("Done!")

if __name__ == "__main__":
    conv_bnorm_loop(32, 0, False, 'nsys')