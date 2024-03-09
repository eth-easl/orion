import torch
from torchvision import models, datasets, transforms
import torch.nn.functional as F
import logging
import utils
from utils.sync_info import BasicSyncInfo, ConcurrentSyncInfo
import time

from utils.sync_control import *



def eval_wrapper(sync_info: BasicSyncInfo, tid: int, model_config, shared_config):
    utils.seed_everything(shared_config['seed'])
    device = torch.device("cuda:0")
    if 'default' in shared_config and shared_config['default']:
        stream = torch.cuda.default_stream(device=device)
    else:
        if isinstance(sync_info, ConcurrentSyncInfo) and sync_info.isolation_level == 'thread':
            stream = torch.cuda.Stream(device=device, priority=-1 if tid == 0 else 0)
        else:
            stream = torch.cuda.Stream(device=device)
    model, optimizer, train_loader, metric_fn = setup(model_config, shared_config, device)
    model.eval()

    num_requests = model_config['num_iterations']
    num_warm_up_reqs = 10


    train_loader_iter = iter(train_loader)

    def eval():
        data, _ = next(train_loader_iter)
        data = data.to(device)
        model(data)

    utils.measure(eval, num_requests, num_warm_up_reqs, model_config['request_rate'], tid, shared_config, stream, sync_info)

def train_wrapper(sync_info: BasicSyncInfo, tid: int, model_config, shared_config):
    utils.seed_everything(shared_config['seed'])
    device = torch.device("cuda:0")

    if 'default' in shared_config and shared_config['default']:
        stream = torch.cuda.default_stream(device=device)
    else:
        if isinstance(sync_info, ConcurrentSyncInfo) and sync_info.isolation_level == 'thread':
            stream = torch.cuda.Stream(device=device, priority=-1 if tid == 0 else 0)
        else:
            stream = torch.cuda.Stream(device=device)

    model, optimizer, train_loader, metric_fn = setup(model_config, shared_config, device)
    model.train()
    num_iterations = model_config['num_iterations']
    logging.info(f'model is set up with num iterations {num_iterations}')

    warm_up_iters = 10

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx == warm_up_iters:
            # finish previous work
            stream.synchronize()
            sync_info.pre_measurement_prep(tid)
            # start timer
            start_time = time.time()

        # if shared_config['enable_profiling'] and batch_idx == warm_up_iters and tid == 0:
        #     torch.cuda.cudart().cudaProfilerStart()

        data, target = batch[0].to(device), batch[1].to(device)
        with ForwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=stream):
            # if shared_config['enable_profiling'] and batch_idx >= warm_up_iters:
            #     torch.cuda.nvtx.range_push(f'thread {tid} starts FORWARD {batch_idx}')
            with torch.cuda.stream(stream):
                output = model(data)
                loss = metric_fn(output, target)
        # if shared_config['enable_profiling'] and batch_idx >= warm_up_iters:
        #     torch.cuda.nvtx.range_pop()

        with BackwardControl(thread_id=tid, batch_idx=batch_idx, sync_info=sync_info, stream=stream):
            # if shared_config['enable_profiling'] and batch_idx >= warm_up_iters:
            #     torch.cuda.nvtx.range_push(f'thread {tid} starts BACKWARD {batch_idx}')
            with torch.cuda.stream(stream):
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        # if shared_config['enable_profiling'] and batch_idx >= warm_up_iters:
        #     torch.cuda.nvtx.range_pop()

        if not sync_info.should_continue_loop(tid, batch_idx, num_iterations):
            break

    stream.synchronize()
    duration = time.time() - start_time
    sync_info.post_measurement_prep(tid)
    sync_info.write_kv(f'duration-{tid}', duration)
    sync_info.write_kv(f'iterations-{tid}', batch_idx + 1)
    sync_info.write_kv(f'throughput-{tid}', (batch_idx-warm_up_iters)/duration)

    logging.info(f'tid {tid} it takes {duration} seconds to train imagenet')
    return duration


def setup(model_config, shared_config, device):
    torch.cuda.set_device(device)
    arch = model_config['arch']
    logging.info(f'vision model with arch {arch}')
    model = models.__dict__[arch](num_classes=1000)
    model = model.to(device)
    # optimizer_func = getattr(torch.optim, model_config['optimizer'])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch_size = model_config['batch_size']
    metric_fn = torch.nn.CrossEntropyLoss().to(device)

    pin_memory = shared_config['pin_memory']

    train_loader = utils.DummyDataLoader(batch=(
        torch.rand([batch_size, 3, 224, 224], pin_memory=pin_memory),
        torch.ones([batch_size], pin_memory=pin_memory).to(torch.long)
    ))
    # else:
    #     if arch == 'inception_v3':
    #         train_transform = transforms.Compose([
    #             transforms.Resize(299),
    #             transforms.CenterCrop(299),
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    #     else:
    #         train_transform = transforms.Compose([
    #             transforms.RandomResizedCrop(224),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    #
    #     train_dataset = \
    #         datasets.ImageFolder(shared_config['imagenet_root'], transform=train_transform)
    #
    #     train_loader = torch.utils.data.DataLoader(
    #         train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return model, optimizer, train_loader, metric_fn
