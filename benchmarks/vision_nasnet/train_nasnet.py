import os
from platform import node
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
import torchvision

from nasnet import NASNetALarge, NASNetAMobile

# preprocessing sources:
# https://github.com/kuangliu/pytorch-cifar10
# https://arxiv.org/pdf/1512.03385.pdf

# code from:
# https://github.com/wandering007/nasnet-pytorch

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training using NasNet')

parser.add_argument('--nas_type', default='large', type=str, help='type of NasNet (large or mobile)')
parser.add_argument('--batchsize', default=128, type=int, help='batch size for training')
parser.add_argument('--optimizer', default='sgd', type=str, help='Optimizer (sgd or Adadelta for now)')
parser.add_argument('--train_dir', default='/cifar/train', type=str, help='cifar10 dataset')
parser.add_argument('--train', action='store_true', help='use model for training')

args = parser.parse_args()

def train():

    print(f"Process with pid {os.getpid()}, args is {args}", args)

    local_rank = 0
    torch.cuda.set_device(local_rank)
    model = NASNetALarge(1001) if args.nas_type == 'large' else NASNetAMobile(1001)
    model = model.to(local_rank) # to GPU


    optimizer =  torch.optim.SGD(model.parameters(), lr=0.1)
    metric_fn = F.cross_entropy

    print("Configure dataset")

    train_dir = args.train_dir
    
    train_transform = transforms.Compose([
                                #transforms.RandomCrop(32, padding=4),
                                #transforms.RandomHorizontalFlip(),
                                transforms.RandomResizedCrop(331),
                                transforms.RandomHorizontalFlip(),
                                #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_dataset = \
            datasets.ImageFolder(train_dir,transform=train_transform)

    train_sampler = torch.utils.data.RandomSampler(
                    train_dataset)
    train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=args.batchsize, sampler=train_sampler, num_workers=8)


    for i in range(1):
        print("Start epoch: ", i)

        if args.train:
            model.train()
        else:
            model.eval()
        
        train_size = len(train_loader)
        print("train size is: ", train_size)

        train_iter = enumerate(train_loader)

        start = time.time()
        start_iter = time.time()

        batch_idx, batch = next(train_iter)

        while batch_idx < 200: #train_size:
            
            if args.train:
                optimizer.zero_grad()
            
            data, target = batch[0].to(local_rank), batch[1].to(local_rank)


            if args.train:
                output = model(data)
            else:
                with torch.no_grad():
                    output = model(data) 
            

            if args.train:
                loss = metric_fn(output, target)
                loss.backward()
                optimizer.step()
            
            print("Iter ", batch_idx, " took ", time.time()-start_iter)
            batch_idx, batch = next(train_iter)

            start_iter = time.time()

        print("Epoch took: ", time.time()-start)

train()
