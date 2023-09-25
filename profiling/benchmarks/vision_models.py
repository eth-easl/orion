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

    data = torch.ones([batchsize, 3, 224, 224], pin_memory=True).to(local_rank)
    target = torch.ones([batchsize], pin_memory=True).to(torch.long).to(local_rank)
    #data = torch.rand([batchsize, 2048]).to(local_rank)
    model = models.__dict__[model_name](num_classes=1000)
    model = model.to(local_rank)

    '''
    train_dir = "/mnt/data/home/fot/imagenet/imagenet-raw-euwest4/"
    train_transform =  transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    train_dataset = \
            datasets.ImageFolder(train_dir,transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=batchsize, num_workers=8)

    train_iter = enumerate(train_loader)
    '''

    if do_eval:
        model.eval()
    else:
        model.train()
        optimizer =  torch.optim.SGD(model.parameters(), lr=0.1)
        criterion =  torch.nn.CrossEntropyLoss().to(local_rank)

    batch_idx = 0
    torch.cuda.synchronize()
    start = time.time()


    for batch_idx in range(1000): #batch in train_iter:

        #data, target = batch[0].to(local_rank), batch[1].to(local_rank)
        start = time.time()
        if batch_idx == 9:
            if profile == 'ncu':
                torch.cuda.nvtx.range_push("start")
            elif profile == 'nsys':
                torch.cuda.profiler.cudart().cudaProfilerStart()
        if do_eval:
            with torch.no_grad():
                output = model(data)
        else:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        if batch_idx == 9:
            if profile == 'ncu':
                torch.cuda.nvtx.range_pop()
            elif profile == 'nsys':
                torch.cuda.profiler.cudart().cudaProfilerStop()
        #batch_idx += 1

        print(f"Iteration took {time.time()-start} sec")

    print(f"Done!, It took {time.time()-start_all} sec")

if __name__ == "__main__":
    vision('mobilenet_v2', 4, 0, True, 'ncu')
