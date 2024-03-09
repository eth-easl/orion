import torch
import threading
import time
import sys

sys.path.append(f"{os.path.expanduser( '~' )}/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch")

from mem_transformer import MemTransformerLM
import lamb

def transformer(batchsize, local_rank, do_eval=True, profile=None):

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

    model = MemTransformerLM(**model_config).to(0)

    if do_eval:
        model.eval()
    else:
        model.train()
        optimizer = lamb.Lamb(model.parameters(), lr=0.1)

    torch.cuda.synchronize()
    batch_idx = 0
    mems = None

    while batch_idx < 10:

        start_iter = time.time()
        if batch_idx == 0:
            if profile == 'ncu':
                torch.cuda.nvtx.range_push("start")
            elif profile == 'nsys':
                torch.cuda.profiler.cudart().cudaProfilerStart()

        if do_eval:
            with torch.no_grad():
                output = model(data, target, mems)
        else:
            optimizer.zero_grad()
            loss, mems = model(data, target, mems)
            loss = loss.float().mean().type_as(loss)
            loss.backward()
            optimizer.step()

        if batch_idx == 9:
            if profile == 'ncu':
                torch.cuda.nvtx.range_pop()
            elif profile == 'nsys':
                torch.cuda.profiler.cudart().cudaProfilerStop()

        batch_idx += 1
        print(f"It took {time.time()-start_iter} sec")

    print("Done!")

if __name__ == "__main__":
    transformer(4, 0, True, 'ncu')
