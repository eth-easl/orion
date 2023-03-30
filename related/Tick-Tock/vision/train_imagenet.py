import torch
from torchvision import models, datasets, transforms
import torch.nn.functional as F
import logging
import utils
import time
from utils.sync_info import SyncInfo
from utils.sync_control import *
import utils.constants as constants


def train_wrapper(my_stream, sync_info: SyncInfo, tid: int, num_epochs: int, device, model_config):
    # model, optimizer, train_loader, metric_fn = setup(model_config, device)
    torch.cuda.set_device(device)
    arc = model_config['arc']
    model = models.__dict__[arc](num_classes=1000)
    model = model.to(device)
    optimizer_func = getattr(torch.optim, model_config['optimizer'])
    optimizer = optimizer_func(model.parameters(), lr=0.1)
    batch_size = model_config['batch_size']
    metric_fn = F.cross_entropy

    model.train()
    num_batches = 300  # len(train_loader)
    logging.info(f'model is set up with num iterations {num_batches}')
    # forward_time = 0
    # backward_time = 0
    # print_every = 10
    warm_up_iters = 30

    data = torch.rand([batch_size, 3, 224, 224], dtype=torch.float).to(device)
    target = torch.randint(high=1000, size=(batch_size,)).to(device)

    with TrainingControl(sync_info=sync_info, device=device):
        start_time = time.time()
        for batch_idx in range(num_batches):

            if constants.enable_profiling and batch_idx == warm_up_iters - 3 and tid == 0:
                torch.cuda.cudart().cudaProfilerStart()

            with ForwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=my_stream):
                # forward_start_time = time.time()
                if constants.enable_profiling and batch_idx >= warm_up_iters:
                    torch.cuda.nvtx.range_push(f'thread {tid} starts FORWARD {batch_idx}')
                with torch.cuda.stream(my_stream):
                    output = model(data)
                    loss = metric_fn(output, target)
                    del output
            if constants.enable_profiling and batch_idx >= warm_up_iters:
                torch.cuda.nvtx.range_pop()
            # forward_time += time.time() - forward_start_time

            with BackwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=my_stream):
                # backward_start_time = time.time()
                if constants.enable_profiling and batch_idx >= warm_up_iters:
                    torch.cuda.nvtx.range_push(f'thread {tid} starts BACKWARD {batch_idx}')
                with torch.cuda.stream(my_stream):
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    del loss
            if constants.enable_profiling and batch_idx >= warm_up_iters:
                torch.cuda.nvtx.range_pop()
            # backward_time += time.time() - backward_start_time

            # if batch_idx % print_every == 0:
            #     logging.info(f'iters {batch_idx}: thread {tid} averaged forward time {forward_time / print_every}; averaged backward time {backward_time / print_every}')
            #     forward_time = 0
            #     backward_time = 0
    torch.cuda.synchronize(device)
    logging.info(f'tid {tid} it takes {time.time() - start_time} seconds to train imagenet')


def setup(model_config, device):
    torch.cuda.set_device(device)
    arc = model_config['arc']
    model = models.__dict__[arc](num_classes=1000)
    model = model.to(device)
    optimizer_func = getattr(torch.optim, model_config['optimizer'])
    optimizer = optimizer_func(model.parameters(), lr=0.1)

    metric_fn = F.cross_entropy

    if arc == 'inception_v3':
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
        datasets.ImageFolder(constants.imagenet_root, transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=model_config['batch_size'], shuffle=True, num_workers=model_config['num_workers'])

    return model, optimizer, train_loader, metric_fn
