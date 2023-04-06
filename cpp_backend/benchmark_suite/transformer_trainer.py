import torch
import threading
import time

from mem_transformer import MemTransformerLM
import lamb
import numpy as np

def transformer_loop(batchsize, train, local_rank, barriers, tid):

    barriers[0].wait()

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

    data = torch.ones((192, batchsize)).to(torch.int64).cuda()
    target = torch.ones((192, batchsize)).to(torch.int64).cuda()
    mems = torch.ones((16, 192, batchsize, 512)).to(torch.int64).cuda()

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
        batch_idx = 0
        torch.cuda.synchronize()

        while batch_idx < 30:
            print(f"submit!, batch_idx is {batch_idx}")
            if train:
                loss, output = model(data, target, mems)
                loss = loss.float().mean().type_as(loss)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    output = model(data, target, mems)

            batch_idx += 1

            print("sent everything!")

            if (batch_idx == 1) and train: # for backward
                barriers[0].wait()

            if batch_idx < 30:
                    barriers[0].wait()

        torch.cuda.synchronize()
        epoch_time = time.time()-start
        timings.append(epoch_time)
        print("Epoch took: ",epoch_time)

    print("Finished! Ready to join!")
