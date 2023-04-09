import torch
import threading
import time

from mem_transformer import MemTransformerLM
import lamb
import numpy as np

class DummyDataLoader():
    def __init__(self, batchsize):
        self.batchsize = batchsize

    def __iter__(self):
        return self

    def __next__(self):
        data = torch.ones((192, self.batchsize)).to(torch.int64)
        target = torch.ones((192, self.batchsize)).to(torch.int64)
        mems = torch.ones((16, 192, self.batchsize, 512)).to(torch.int64)

        return data, target, mems

def transformer_loop(batchsize, train, num_iters, rps, dummy_data, local_rank, barriers, tid):

    barriers[0].wait()

    if rps > 0:
        sleep_times = np.random.exponential(scale=1/rps, size=num_iters)
    else:
        sleep_times = [0]*num_iters

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

    torch.cuda.profiler.cudart().cudaProfilerStart()

    model = MemTransformerLM(**model_config).to(0)

    if train:
        model.train()
        optimizer = lamb.Lamb(model.parameters(), lr=0.1)
    else:
        model.eval()

    timings=[]
    for i in range(1):
        print("Start epoch: ", i)

        start = time.time()
        start_iter = time.time()
        torch.cuda.synchronize()

        while batch_idx < num_iters:
            print(f"submit!, batch_idx is {batch_idx}")
            if train:
                data, target, mems = batch[0].to(local_rank), batch[1].to(local_rank), batch[2].to(local_rank)
                loss, output = model(data, target, mems)
                loss = loss.float().mean().type_as(loss)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    data, target, mems = batch[0].to(local_rank), batch[1].to(local_rank), batch[2].to(local_rank)
                    output = model(data, target, mems)

            time.sleep(sleep_times[batch_idx])
            print(f"{batch_idx} submitted! sent everything, sleep for {sleep_times[batch_idx]} sec")

            batch_idx, batch = next(train_iter)
            if (batch_idx == 1): # for backward
                barriers[0].wait()

            # if batch_idx < num_iters:
            #         barriers[0].wait()

        epoch_time = time.time()-start
        timings.append(epoch_time)
        print("Epoch took: ",epoch_time)

    print("Finished! Ready to join!")
