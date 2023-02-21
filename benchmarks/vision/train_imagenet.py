""" Basically most vision models. """

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

# preprocessing sources:
# https://github.com/pytorch/examples/blob/main/imagenet/main.py

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training or inference using torchvision models')

parser.add_argument('--arch', default='resnet18', type=str, help='torchvision model')
parser.add_argument('--batchsize', default=128, type=int, help='batch size for training')
parser.add_argument('--optimizer', default='sgd', type=str, help='Optimizer (sgd, adadelta, or adam for now)')
parser.add_argument('--train_dir', default='/mnt/data/home/fot/imagenet/imagenet-raw-euwest4/train', type=str,
                    help='path to ImageNet dataset')
parser.add_argument('--train', action='store_true', help='use model for training')

args = parser.parse_args()


def train():
    print(f"Process with pid {os.getpid()}, args is {args}")

    local_rank = 0
    torch.cuda.set_device(local_rank)
    model = models.__dict__[args.arch](num_classes=1000)
    model = model.to(local_rank)  # to GPU

    if args.train:

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        elif args.optimizer == 'adadelta':
            optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1)
        else:
            print("Optimizer is not supported!")
            return

        metric_fn = torch.nn.CrossEntropyLoss().to(0)

    print("Configure dataset")

    train_dir = args.train_dir

    if args.arch == 'inception_v3':
        train_transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_dataset = \
        datasets.ImageFolder(train_dir, transform=train_transform)

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

        while batch_idx < 200:

            if args.train:
                optimizer.zero_grad()
            data, target = batch[0].to(local_rank), batch[1].to(local_rank)

            if args.train:
                if args.arch == 'inception_v3':
                    output, _ = model(data)
                else:
                    output = model(data)

            else:
                with torch.no_grad():
                    if args.arch == 'inception_v3':
                        output, _ = model(data)
                    else:
                        output = model(data)

            if args.train:
                loss = metric_fn(output, target)
                loss.backward()
                optimizer.step()

            print("Iter ", batch_idx, " took ", time.time() - start_iter)
            batch_idx, batch = next(train_iter)

            start_iter = time.time()

        print("Epoch took: ", time.time() - start)


train()
