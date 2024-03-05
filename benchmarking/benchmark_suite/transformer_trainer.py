import torch
import threading
import time

from mem_transformer import MemTransformerLM
import lamb
import numpy as np
from ctypes import *
import os
import json

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
        self.data = torch.ones((192, self.batchsize), pin_memory=True).to(torch.int64)
        self.target = torch.ones((192, self.batchsize), pin_memory=True).to(torch.int64)

    def __iter__(self):
        return self

    def __next__(self):
        return self.data, self.target


def block(backend_lib, it):
    # block client until request served
    backend_lib.block(it)

def check_stop(backend_lib):
    return backend_lib.stop()


def transformer_loop(batchsize, train, num_iters, rps, uniform, dummy_data, local_rank, barriers, client_barrier, tid, input_file=''):

    seed_everything(42)

    backend_lib = cdll.LoadLibrary(os.path.expanduser('~') + "/orion/src/cuda_capture/libinttemp.so")

    if rps > 0 and input_file=='':
        if uniform:
            sleep_times = [1/rps]*num_iters
        else:
            sleep_times = np.random.exponential(scale=1/rps, size=num_iters)
    elif input_file != '':
        with open(input_file) as f:
            sleep_times = json.load(f)
    else:
        sleep_times = [0] * num_iters

    barriers[0].wait()

    if (train and tid==1):
        time.sleep(10)

    model_config = {
        'n_token': 267735,
        'n_layer': 16,
        'n_head': 8,
        'd_model': 512,
        'd_head': 64,
        'd_inner': 2048,
        'dropout': 0.1,
        'dropatt': 0.0,
        'dtype': None,
        'tie_weight': True,
        'd_embed': 512,
        'div_val': 1,
        'tie_projs': [False, True, True, True],
        'pre_lnorm': False,
        'tgt_len': 192,
        'ext_len': 0,
        'mem_len': 192,
        'cutoffs': [19997, 39997, 199997],
        'same_length': False,
        'attn_type': 0,
        'clamp_len': -1,
        'sample_softmax': -1
    }

    train_loader = DummyDataLoader(batchsize)
    train_iter = enumerate(train_loader)
    batch_idx, batch = next(train_iter)


    model = MemTransformerLM(**model_config).to(0)

    if train:
        model.train()
        optimizer = lamb.Lamb(model.parameters(), lr=0.1)
    else:
        model.eval()

    mems = None

    #  open loop
    next_startup = time.time()
    open_loop = True

    for i in range(1):
        print("Start epoch: ", i)

        start = time.time()
        start_iter = time.time()
        timings = []
        while batch_idx < num_iters:
            if train:
                print(f"Client {tid}, start iter {batch_idx}")
                data, target = batch[0].to(local_rank), batch[1].to(local_rank)
                #optimizer.zero_grad()
                loss, mems = model(data, target, mems)
                loss = loss.float().mean().type_as(loss)
                loss.backward()
                optimizer.step()
                block(backend_lib, batch_idx)
                batch_idx, batch = next(train_iter)
                if (batch_idx == 1): # for backward
                    barriers[0].wait()
                if batch_idx == 10:
                    barriers[0].wait()
                    print("After sync!!")
                    start = time.time()
                if check_stop(backend_lib):
                    print("---- STOP!")
                    break
            else:
                with torch.no_grad():
                    cur_time = time.time()
                    #### OPEN LOOP ####
                    if open_loop:
                        if (cur_time >= next_startup):
                            print(f"Client {tid}, submit!, batch_idx is {batch_idx}")
                            if batch_idx==50:
                                torch.cuda.profiler.cudart().cudaProfilerStart()
                            data, target = batch[0].to(local_rank), batch[1].to(local_rank)
                            output, mems = model(data, target, mems)
                            block(backend_lib, batch_idx)
                            req_time = time.time()-next_startup
                            timings.append(req_time)
                            print(f"Client {tid} finished! Wait! It took {req_time}")
                            if batch_idx>=10:
                                next_startup += sleep_times[batch_idx]
                            else:
                                next_startup = time.time()
                            batch_idx,batch = next(train_iter)
                            if (batch_idx == 1 or (batch_idx == 10)):
                                    barriers[0].wait()
                                    # hp starts after
                                    if (batch_idx==10):
                                        next_startup = time.time()
                                        start = time.time()
                            dur = next_startup-time.time()
                            if (dur>0):
                                time.sleep(dur)
                            if check_stop(backend_lib):
                                print("---- STOP!")
                                break
                    else:
                        #### CLOSED LOOP ####
                        print(f"Client {tid}, submit!, batch_idx is {batch_idx}")
                        data, target = batch[0].to(local_rank), batch[1].to(local_rank)
                        output, mems = model(data, target, mems)
                        print(f"Client {tid} finished! Wait!")
                        batch_idx,batch = next(train_iter)
                        if ((batch_idx == 1) or (batch_idx == 10)):
                            barriers[0].wait()


    barriers[0].wait()
    total_time = time.time() - start

    if not train and len(timings)>0:
        print(timings)
        p50 = np.percentile(timings, 50)
        p95 = np.percentile(timings, 95)
        p99 = np.percentile(timings, 99)
        print(f"Client {tid} finished! p50: {p50} sec, p95: {p95} sec, p99: {p99} sec")
        data = {
            'p50_latency': p50*1000,
            'p95_latency': p95*1000,
            'p99_latency': p99*1000,
            'throughput': (batch_idx-10)/total_time
        }
    else:
        data = {
            'throughput': (batch_idx-10)/total_time
        }
    with open(f'client_{tid}.json', 'w') as f:
        json.dump(data, f)

    print("Finished! Ready to join!")
