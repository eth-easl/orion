import argparse
import json
import threading
import multiprocessing
import time
from ctypes import *
import os
import sys
from torchvision import models
import torch
import numpy as np

torch.set_num_threads(2)

# sys.path.insert(0, "/home/image-varuna/DeepLearningExamples/PyTorch/Translation/GNMT")
# from benchmark_suite.gnmt_trainer import gnmt_loop
sys.path.append("/home/image-varuna/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch")
sys.path.append("/home/image-varuna/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch/utils")
from benchmark_suite.transformer_trainer_torch import transformer_loop
sys.path.append("/home/image-varuna/DeepLearningExamples/PyTorch/LanguageModeling/BERT")
from bert_trainer_torch import bert_loop
sys.path.append("/home/image-varuna/mlcommons/single_stage_detector/ssd")
from benchmark_suite.retinanet_trainer_torch import retinanet_loop
sys.path.append("/home/image-varuna/DeepLearningExamples/PyTorch/Recommendation/DLRM")
#from benchmark_suite.dlrm_trainer import dlrm_loop

from benchmark_suite.train_imagenet_torch import imagenet_loop
from scheduler_frontend import PyScheduler


function_dict = {
    "resnet50": imagenet_loop,
    "resnet101": imagenet_loop,
    "mobilenet_v2": imagenet_loop,
    "efficientnet": imagenet_loop,
    "bert": bert_loop,
    "gnmt": None, #gnmt_loop,
    "transformer": transformer_loop,
    "retinanet": retinanet_loop,
}

def seed_everything(seed: int):
    import random, os
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def launch_jobs(config_dict_list, mode, processes):

    print(config_dict_list)
    num_clients = len(config_dict_list)
    print(num_clients)

    # init
    if processes:
        start_barriers = [multiprocessing.Barrier(num_clients+1) for i in range(num_clients)]
        end_barriers = [multiprocessing.Barrier(num_clients+1) for i in range(num_clients)]
    else:
        start_barriers = [threading.Barrier(num_clients+1) for i in range(num_clients)]
        end_barriers = [threading.Barrier(num_clients+1) for i in range(num_clients)]

    print(torch.__version__)

    model_names = [config_dict['arch'] for config_dict in config_dict_list]
    model_files = [config_dict['kernel_file'] for config_dict in config_dict_list]
    num_kernels = [config_dict['num_kernels'] for config_dict in config_dict_list]
    num_iters = [config_dict['num_iters'] for config_dict in config_dict_list]

    tids = []
    threads = []
    for i, config_dict in enumerate(config_dict_list):
        func = function_dict[config_dict['arch']]
        model_args = config_dict['args']
        default = True if mode=="sequential" else False
        model_args.update({"num_iters":num_iters[i], "default": default, "local_rank": 0, "start_barriers": start_barriers, "end_barriers": end_barriers, "tid": i})

        if processes:
            thread = multiprocessing.Process(target=func, kwargs=model_args)
        else:
            thread = threading.Thread(target=func, kwargs=model_args)

        thread.start()
        #tids.append(thread.native_id)
        threads.append(thread)

    print(tids)

    print("before starting")



    timings=[]
    # if mode == "sequential":
    #     print("wait!")
    #     start_barriers[0].wait()
    #     start_barriers[1].wait()

    #     start_barriers[0].wait()
    #     start = time.time()
    #     end_barriers[0].wait()
    #     torch.cuda.synchronize()
    #     print(f"Client 0 took {time.time()-start} sec")

    #     start_barriers[1].wait()
    #     startB = time.time()
    #     end_barriers[1].wait()
    #     torch.cuda.synchronize()
    #     print(f"Client 1 took {time.time()-startB}")

    #     total_time = time.time()-start
    #     print(f"Time for both is {time.time()-start} sec")


    # elif mode == "streams":

    start_barriers[0].wait()
    #start_barriers[1].wait()

    torch.cuda.profiler.cudart().cudaProfilerStart()
    start = time.time()

    #for i in range(50):
    #    start_barriers[0].wait()
        #start_barriers[1].wait()
        #end_barriers[0].wait()
        #end_barriers[1].wait()
    end_barriers[0].wait()
    torch.cuda.synchronize()
    torch.cuda.profiler.cudart().cudaProfilerStop()
    print(f"Time for both is {time.time()-start} sec")

    # for i in range(10):
    #     start = time.time()
    #     print(f"start iter {i}")
    #     start_barriers[0].wait()
    #     start_barriers[1].wait()

    #     end_barriers[0].wait()
    #     end_barriers[1].wait()
    #     torch.cuda.synchronize()
    #     # print(f"Part A took {time.time()-start}")
    #     # startB = time.time()
    #     # start_barriers[1].wait()
    #     # end_barriers[1].wait()
    #     # torch.cuda.synchronize()
    #     total_time = time.time()-start
    #     #print(f"Part B took {time.time()-startB}")
    #     print(f"Iteration {i} took {total_time} sec")
    #     timings.append(total_time)

    # timings = timings[2:]
    # print(f"Avg is {np.median(np.asarray(timings))} sec, Min is {min(timings)} sec")

    for thread in threads:
        thread.join()

    print("train joined!")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    config_file = sys.argv[1]
    mode = sys.argv[2] # "sequential" or "streams"
    processes = False
    with open(config_file) as f:
        config_dict = json.load(f)
    launch_jobs(config_dict, mode, processes)
