import torch
import threading
import time

from mem_transformer import MemTransformerLM
import lamb
import numpy as np

class DummyDataLoader():
    def __init__(self, batchsize):
        self.batchsize = batchsize
        self.data = torch.ones((192, self.batchsize), pin_memory=True).to(torch.int64)
        self.target = torch.ones((192, self.batchsize), pin_memory=True).to(torch.int64)

    def __iter__(self):
        return self

    def __next__(self):
        return self.data, self.target

def transformer_loop(batchsize, train, default, num_iters, rps, uniform, dummy_data, local_rank, start_barriers, end_barriers, tid):

    start_barriers[0].wait()

    if rps > 0:
        if uniform:
            sleep_times = [1/rps]*num_iters
        else:
            sleep_times = np.random.exponential(scale=1/rps, size=num_iters)
    else:
        sleep_times = [0]*num_iters

    if default:
        s = torch.cuda.default_stream()
    else:
        s = torch.cuda.Stream()
    timings = []

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

    next_startup = time.time()
    open_loop = False
    timings = [0 for _ in range(num_iters)]

    mems = None
    print("before while")
    with torch.cuda.stream(s):
        for i in range(1):
            print("Start epoch: ", i)

            while batch_idx < num_iters:
                start = time.time()

                if train:
                    optimizer.zero_grad()
                    start_iter = time.time()
                    data, target = batch[0].to(local_rank), batch[1].to(local_rank)
                    loss, mems = model(data, target, mems)
                    loss = loss.float().mean().type_as(loss)
                    loss.backward()
                    optimizer.step()
                    #s.synchronize()
                    print(f"Client {tid}, iter {batch_idx} took {time.time()-start_iter} sec")
                    batch_idx,batch = next(train_iter)
                    if (batch_idx==10):
                        starttime = time.time()
                else:
                    with torch.no_grad():
                        cur_time = time.time()
                        ###### OPEN LOOP #####
                        if (cur_time >= next_startup):
                            print(f"Client {tid} submit!, batch_idx is {batch_idx}")
                            data, target = batch[0].to(local_rank), batch[1].to(local_rank)
                            output, mems = model(data, target, mems)
                            s.synchronize()
                            timings[batch_idx] = time.time()-next_startup
                            print(f"It took {timings[batch_idx]} sec")
                            next_startup += sleep_times[batch_idx]
                            batch_idx,batch = next(train_iter)
                            if (batch_idx==10):
                                starttime = time.time()

            print(f"FINISHED! It took {time.time()-starttime} sec")
            end_barriers[0].wait()

    #print(f"Time is {time.time()-starttime} sec")
    if not train:
        timings = timings[2:]
        p50 = np.percentile(timings, 50)
        p95 = np.percentile(timings, 95)
        p99 = np.percentile(timings, 99)

        print(f"Client {tid} finished! p50: {p50} sec, p95: {p95} sec, p99: {p99} sec")
        print(f"Total time is {time.time()-starttime} sec")
