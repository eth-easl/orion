import torch
from torchvision import models, datasets, transforms
import torch.nn.functional as F
import logging
import utils
import time
from utils.sync_control import *



def train_wrapper(sync_info, tid: int, model_config, shared_config):
    device = torch.device("cuda:0")
    my_stream = torch.cuda.Stream(device=device)
    model, optimizer, train_loader, metric_fn = setup(model_config, shared_config, device)
    model.train()
    num_iterations = model_config['num_iterations']
    logging.info(f'model is set up with num iterations {num_iterations}')

    warm_up_iters = model_config['warm_up_iters']

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx == warm_up_iters:
            # finish previous work
            torch.cuda.synchronize(device)
            sync_info.pre_measurement_prep(tid)
            # start timer
            start_time = time.time()

        # if shared_config['enable_profiling'] and batch_idx == warm_up_iters and tid == 0:
        #     torch.cuda.cudart().cudaProfilerStart()

        data, target = batch[0].to(device), batch[1].to(device)
        with ForwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=my_stream):
            # if shared_config['enable_profiling'] and batch_idx >= warm_up_iters:
            #     torch.cuda.nvtx.range_push(f'thread {tid} starts FORWARD {batch_idx}')
            with torch.cuda.stream(my_stream):
                output = model(data)
                loss = metric_fn(output, target)
        # if shared_config['enable_profiling'] and batch_idx >= warm_up_iters:
        #     torch.cuda.nvtx.range_pop()

        with BackwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=my_stream):
            # if shared_config['enable_profiling'] and batch_idx >= warm_up_iters:
            #     torch.cuda.nvtx.range_push(f'thread {tid} starts BACKWARD {batch_idx}')
            with torch.cuda.stream(my_stream):
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        # if shared_config['enable_profiling'] and batch_idx >= warm_up_iters:
        #     torch.cuda.nvtx.range_pop()

        if batch_idx == num_iterations - 1:
            # reached the last iteration
            break
    sync_info.no_sync_control = True
    torch.cuda.synchronize(device)
    sync_info.post_measurement_prep(tid)
    duration = time.time() - start_time
    logging.info(f'tid {tid} it takes {duration} seconds to train imagenet')
    return duration


def setup(model_config, shared_config, device):
    torch.cuda.set_device(device)
    arc = model_config['arc']
    model = models.__dict__[arc](num_classes=1000)
    model = model.to(device)
    optimizer_func = getattr(torch.optim, model_config['optimizer'])
    optimizer = optimizer_func(model.parameters(), lr=0.1)
    batch_size = model_config['batch_size']
    metric_fn = F.cross_entropy

    if shared_config['use_dummy_data']:
        train_loader = utils.DummyDataLoader(
            batch=(
                torch.rand([batch_size, 3, 224, 224], dtype=torch.float).to(device),
                torch.randint(high=1000, size=(batch_size,)).to(device)
            )
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
            datasets.ImageFolder(shared_config['imagenet_root'], transform=train_transform)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=model_config['num_workers'])

    return model, optimizer, train_loader, metric_fn
