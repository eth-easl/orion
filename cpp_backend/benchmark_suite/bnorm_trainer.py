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
        self.bn = torch.nn.BatchNorm2d(256)

    def forward(self, x):
        for i in range(25):
            y = self.bn(x)


def bnorm_loop(batchsize, train, local_rank, barriers, tid):

    print(batchsize, local_rank, barriers, tid)
    barriers[0].wait()

    data = torch.rand([batchsize, 256, 112, 112]).to(local_rank).contiguous()
    model = Model()
    model = model.to(0)

    if train:
        model.train()
    else:
        model.eval()

    for i in range(10):
        print("Start epoch: ", i)

        start = time.time()
        start_iter = time.time()

        batch_idx = 0

        while batch_idx < 1:

            print(f"submit!, batch_idx is {batch_idx}")

            if train:
                output = model(data)
            else:
                with torch.no_grad():
                    output = model(data)


            batch_idx += 1

            start_iter = time.time()

        # barriers[0].wait()
        if i < 9:
            barriers[0].wait()
        print(f"{tid}, Epoch done!")

    print("Finished! Ready to join!")
