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
import psutil
import argparse
import threading

def seed_everything(seed: int):
    import random, os
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class DummyDataLoader():
    def __init__(self, batchsize):
        self.batchsize = batchsize
        self.data = torch.rand([self.batchsize, 3, 224, 224], pin_memory=True)
        self.target = torch.ones([self.batchsize], pin_memory=True, dtype=torch.long)

    def __iter__(self):
        return self

    def __next__(self):
        return self.data, self.target

class RealDataLoader():
    def __init__(self, batchsize):
        train_transform =  transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))]
        )
        train_dataset = \
                datasets.ImageFolder("/mnt/data/home/fot/imagenet/imagenet-raw-euwest4",transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batchsize, num_workers=12)

    def __iter__(self):
        print("Inside iter")
        return iter(self.train_loader)

def imagenet_loop(model_name, batchsize, train, default, num_iters, rps, uniform, dummy_data, local_rank, start_barriers, end_barriers, tid):

    seed_everything(42)
    #os.sched_setaffinity(0, {tid+4})
    start_barriers[0].wait()

    if rps > 0:
        if uniform:
            sleep_times = [1/rps]*num_iters
        else:
            sleep_times = np.random.exponential(scale=1/rps, size=num_iters)
    else:
        sleep_times = [0]*num_iters

    print(sleep_times)

    if default:
        s = torch.cuda.default_stream()
    else:
        s = torch.cuda.Stream()
    print("-------------- thread id:  ", threading.get_native_id())

    #data = torch.rand([batchsize, 2048]).to(local_rank)
    model = models.__dict__[model_name](num_classes=1000)
    model = model.to(0)

    if train:
        model.train()
        optimizer =  torch.optim.SGD(model.parameters(), lr=0.1)
        criterion =  torch.nn.CrossEntropyLoss().to(local_rank)
    else:
        model.eval()

    if dummy_data:
        train_loader = DummyDataLoader(batchsize)
    else:
        train_loader = RealDataLoader(batchsize)

    train_iter = enumerate(train_loader)
    print(tid, train_iter)

    batch_idx, batch = next(train_iter)

    gpu_data, gpu_target = batch[0].to(local_rank), batch[1].to(local_rank)
    output = model(gpu_data)

    start_times = [0 for _ in range(num_iters)]
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    cvs = [threading.Event() for _ in range(num_iters)]
    timings = [0 for _ in range(num_iters)]

    timing_idx = 0

    print("Enter loop!")

    next_startup = time.time()

    with torch.cuda.stream(s):
        for i in range(1):
            print("Start epoch: ", i)

            start = time.time()
            while batch_idx < num_iters:
                # #start_barriers[0].wait()
                # startiter = time.time()
                if train:
                    print("here")
                    #start_barriers[0].wait()
                    start_iter = time.time()
                    optimizer.zero_grad()
                    gpu_data, gpu_target = batch[0].to(local_rank), batch[1].to(local_rank)
                    output = model(gpu_data)
                    loss = criterion(output, gpu_target)
                    loss.backward()
                    optimizer.step()
                    #s.synchronize()
                    print(f"Client {tid}, iter {batch_idx} took {time.time()-start_iter} sec")
                    batch_idx,batch = next(train_iter)
                    #end_barriers[0].wait()
                    if (batch_idx==10):
                        starttime = time.time()
                    if batch_idx == 300:
                        print(f"---------- Client {tid} finished! total time is {time.time()-starttime}")

                else:
                    with torch.no_grad():
                        cur_time = time.time()
                        ###### OPEN LOOP #####
                        if (cur_time >= next_startup):
                            gpu_data = batch[0].to(local_rank)
                            output = model(gpu_data)
                            s.synchronize()
                            timings[batch_idx] = time.time()-next_startup
                            print(f"Client {tid}, request {batch_idx} took {timings[batch_idx]} sec")
                            batch_idx,batch = next(train_iter)
                            next_startup += sleep_times[batch_idx-1]
                            dur = next_startup-time.time()
                            if dur > 0:
                                time.sleep(dur)
                            if (batch_idx==10):
                                starttime = time.time()
                                next_startup = time.time()

                        ###### CLOSED LOOP #####
                        # print(f"submit!, batch_idx is {batch_idx}")
                        # gpu_data = batch[0].to(local_rank)
                        # output = model(gpu_data)
                        # s.synchronize()
                        # iter_time = time.time()-startiter
                        # timings.append(iter_time)
                        # print(f"It took {iter_time} sec")
                        # batch_idx,batch = next(train_iter)

                #iter_time = time.time()-startiter
                #timings.append(iter_time)

                #time.sleep(sleep_times[batch_idx])
                #print(f"{batch_idx} finished, took {iter_time} sec, now sleep for {sleep_times[batch_idx]} sec")

                #v = time.time()

                #batch_idx += 1
                #print(f"It took {time.time()-v}")

                # if batch_idx < num_iters-1:
                #     start_barriers[tid].wait()


    timings = timings[10:]
    p50 = np.percentile(timings, 50)
    p95 = np.percentile(timings, 95)
    p99 = np.percentile(timings, 99)

    #end_barriers[0].wait()
    print(f"Client {tid} finished! p50: {p50} sec, p95: {p95} sec, p99: {p99} sec")
    print(f"Total time is {time.time()-starttime} sec")
