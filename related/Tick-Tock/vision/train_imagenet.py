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
    model, optimizer, train_loader, metric_fn = setup(model_config, device)
    model.train()
    num_iterations = model_config['num_iterations']
    logging.info(f'model is set up with num iterations {num_iterations}')

    warm_up_iters = model_config['warm_up_iters']

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx == warm_up_iters:
            # finish previous work
            torch.cuda.synchronize(device)
            if not sync_info.no_sync_control:
                sync_info.barrier.wait()
            # start timer
            start_time = time.time()

        # if constants.enable_profiling and batch_idx == warm_up_iters and tid == 0:
        #     torch.cuda.cudart().cudaProfilerStart()

        data, target = batch[0].to(device), batch[1].to(device)
        with ForwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=my_stream):
            # if constants.enable_profiling and batch_idx >= warm_up_iters:
            #     torch.cuda.nvtx.range_push(f'thread {tid} starts FORWARD {batch_idx}')
            with torch.cuda.stream(my_stream):
                output = model(data)
                loss = metric_fn(output, target)
        # if constants.enable_profiling and batch_idx >= warm_up_iters:
        #     torch.cuda.nvtx.range_pop()

        with BackwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=my_stream):
            # if constants.enable_profiling and batch_idx >= warm_up_iters:
            #     torch.cuda.nvtx.range_push(f'thread {tid} starts BACKWARD {batch_idx}')
            with torch.cuda.stream(my_stream):
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        # if constants.enable_profiling and batch_idx >= warm_up_iters:
        #     torch.cuda.nvtx.range_pop()

        if batch_idx == num_iterations - 1:
            # reached the last iteration
            break

    torch.cuda.synchronize(device)
    logging.info(f'tid {tid} it takes {time.time() - start_time} seconds to train imagenet')


def setup(model_config, device):
    torch.cuda.set_device(device)
    arc = model_config['arc']
    model = models.__dict__[arc](num_classes=1000)
    model = model.to(device)
    optimizer_func = getattr(torch.optim, model_config['optimizer'])
    optimizer = optimizer_func(model.parameters(), lr=0.1)
    batch_size = model_config['batch_size']
    metric_fn = F.cross_entropy

    if constants.use_dummy_data:
        train_loader = utils.DummyDataLoader(
            data=torch.rand([batch_size, 3, 224, 224], dtype=torch.float).to(device),
            target=torch.randint(high=1000, size=(batch_size,)).to(device),
            iterations=model_config['num_iterations']
        )
    else:
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
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=model_config['num_workers'])

    return model, optimizer, train_loader, metric_fn
