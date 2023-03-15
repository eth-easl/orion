import torch
import threading
import time

from seq2seq.models.gnmt import GNMT

def gnmt_loop(batchsize):


    model_config = {
        "hidden_size": 1024,
        "vocab_size": 32320,
        "num_layers": 4,
        "dropout": 0.2,
        "batch_first": False,
        "share_embedding": True 
    }

    print("-------------- thread id:  ", threading.get_native_id())

    input0 = torch.ones([50, batchsize]).to(torch.int64).to(0)
    input1 = torch.ones([batchsize]).to(torch.int64).to(0) 
    input2 = torch.ones([50, batchsize]).to(torch.int64).to(0) 

    model = GNMT(**model_config).to(0)

    model.eval()


    for i in range(1):
        print("Start epoch: ", i)

        start = time.time()
        start_iter = time.time()
        batch_idx = 0
        torch.cuda.synchronize()
        
        while batch_idx < 1:
            print(f"submit!, batch_idx is {batch_idx}")
            torch.cuda.profiler.cudart().cudaProfilerStart()
            
            with torch.no_grad():
                output = model(input0, input1, input2)
        
            torch.cuda.profiler.cudart().cudaProfilerStop()
                                                           
            print(output.shape)
            batch_idx += 1

            start_iter = time.time()
                                                                                                                                                                                            
    print("Epoch took: ", time.time()-start)

print(torch.__version__)
gnmt_loop(64)
                                                                                                                                                                                                                                                                                                             
