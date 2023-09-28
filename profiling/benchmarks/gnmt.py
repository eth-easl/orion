import torch
import threading
import time
import sys

sys.path.insert(0, f"{os.path.expanduser( '~' )}/DeepLearningExamples/PyTorch/Translation/GNMT")


from seq2seq.models.gnmt import GNMT

def gnmt(batchsize, local_rank, do_eval=True, profile=None):

    model_config = {

        "hidden_size": 1024,
        "vocab_size": 32320,
        "num_layers": 4,
        "dropout": 0.2,
        "batch_first": False,
        "share_embedding": True
    }

    input0 = torch.ones([50, batchsize]).to(torch.int64).to(0)
    input1 = torch.ones([batchsize]).to(torch.int64).to(0)
    input2 = torch.ones([50, batchsize]).to(torch.int64).to(0)
    labels = input2

    model = GNMT(**model_config).to(local_rank)

    if do_eval:
        model.eval()
    else:
        model.train()
        #criterion = LabelSmoothing(0.1, 0).to(local_rank)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    batch_idx = 0
    torch.cuda.synchronize()

    while batch_idx < 10:

        #if not do_eval:
        #    optimizer.zero_grad()

        if batch_idx == 0:
            if profile == 'ncu':
                torch.cuda.nvtx.range_push("start")
            elif profile == 'nsys':
                torch.cuda.profiler.cudart().cudaProfilerStart()

        if do_eval:
            with torch.no_grad():
                output = model(input0, input1, input2)
        else:
            output = model(input0, input1, input2)
            #T, B = output.size(0), output.size(1)
            #loss = criterion(output.view(T * B, -1), labels.contiguous().view(-1))
            #loss.backward()
            #optimizer.step()


        if batch_idx == 9:
            if profile == 'ncu':
                torch.cuda.nvtx.range_pop()
            elif profile == 'nsys':
                torch.cuda.profiler.cudart().cudaProfilerStop()

        batch_idx += 1

        print("Done!")

if __name__ == "__main__":
    gnmt(128, 0, False, 'nsys')
