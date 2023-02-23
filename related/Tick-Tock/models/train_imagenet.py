import torch
from torchvision import models, datasets, transforms
import torch.nn.functional as F
from datetime import datetime

from src.train_info import TrainInfo
from src.sync_info import SyncInfo
from src.sync_controller import *
import time

def pretty_time():
    return datetime.now().strftime('%H:%M:%S')


def train_wrapper(my_stream, sync_info: SyncInfo, tid: int, num_epochs : int, device, train_info: TrainInfo):
    model, optimizer, train_loader, metric_fn = setup(train_info, device)
    model.train()
    print(f"training {tid} starts!!")

    start = time.time()
    loss_sum = 0
    print_every = 50
    for epoch in range(num_epochs):

        train_size = len(train_loader)
        print("train size is: ", train_size)

        for batch_idx, batch in enumerate(train_loader):
            with ForwardController(thread_id=tid, sync_info=sync_info):
                print(f"time: {pretty_time()}, thread {tid} starts FORWARD {batch_idx}")
                with torch.cuda.stream(my_stream):
                    data, target = batch[0].to(device), batch[1].to(device)
                    output = model(data)
                    loss = metric_fn(output, target)
                    loss_sum += loss.item()
                print(f"time: {pretty_time()}, thread {tid} ends FORWARD {batch_idx}")

            if batch_idx % print_every == 0:
                print(f"loss for thread {tid}: {loss_sum / print_every}")
                loss_sum = 0

            with BackwardController(thread_id=tid, sync_info=sync_info):
                print(f"time: {pretty_time()}, thread {tid} starts BACKWARD {batch_idx}")
                with torch.cuda.stream(my_stream):
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                print(f"time: {pretty_time()}, thread {tid} ends BACKWARD {batch_idx}")
    sync_info.no_sync_control = True
    end = time.time()
    print(f"TID: {tid}, training took {end - start} sec.")


def setup(train_info: TrainInfo, device):
    torch.cuda.set_device(device)
    model = models.__dict__[train_info.arch](num_classes=1000)
    model = model.to(device)
    optimizer_func = getattr(torch.optim, train_info.optimizer)
    optimizer = optimizer_func(model.parameters(), lr=0.1)

    metric_fn = F.cross_entropy

    if train_info.arch == 'inception_v3':
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
        datasets.ImageFolder(train_info.train_dir, transform=train_transform)

    train_sampler = torch.utils.data.RandomSampler(
        train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_info.batchsize, sampler=train_sampler, num_workers=train_info.num_workers)

    return model, optimizer, train_loader, metric_fn
