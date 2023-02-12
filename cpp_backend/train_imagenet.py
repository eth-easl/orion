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

def imagenet_loop(parent_model, batchsize, train_loader, local_rank, barrier, tid):

    # do only forward for now, experimental
    barrier.wait()

    print("-------------- thread id:  ", threading.get_native_id())

    data = torch.rand([batchsize, 3, 224, 224]).to(local_rank)
    model = models.__dict__['resnet50'](num_classes=1000)
    model = model.to(0)

    model.eval()

    
    for i in range(1):
        print("Start epoch: ", i)

        start = time.time()
        start_iter = time.time()

        batch_idx = 0

        torch.cuda.synchronize()

        while batch_idx < 1:
 
            print(f"submit!, batch_idx is {batch_idx}")
            with torch.no_grad():
                output = model(data)
            
            #print(output)
            batch_idx += 1

            start_iter = time.time()

            # notify the scheduler that everything is submitted
            #barrier.wait()
            
            #torch.cuda.synchronize()
            # wait until all operations have been completed before starting the next iter
            #barrier.wait()
        
        print("Epoch took: ", time.time()-start)
        while(True):
            pass
