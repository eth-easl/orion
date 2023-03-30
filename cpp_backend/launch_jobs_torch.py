import argparse
import json
import threading
import time
from ctypes import *
import os
import sys
from torchvision import models
import torch
import numpy as np

# sys.path.insert(0, "/home/image-varuna/DeepLearningExamples/PyTorch/Translation/GNMT")
# from benchmark_suite.gnmt_trainer import gnmt_loop
# sys.path.append("/home/image-varuna/DeepLearningExamples/PyTorch/LanguageModeling/BERT")
# from benchmark_suite.bert_trainer import bert_loop
sys.path.append("/home/image-varuna/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch")
sys.path.append("/home/image-varuna/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch/utils")
from benchmark_suite.transformer_trainer import transformer_loop
sys.path.append("/home/image-varuna/mlcommons/single_stage_detector/ssd")
#from benchmark_suite.retinanet_trainer import retinanet_loop
sys.path.append("/home/image-varuna/DeepLearningExamples/PyTorch/Recommendation/DLRM")
#from benchmark_suite.dlrm_trainer import dlrm_loop

from benchmark_suite.train_imagenet_torch import imagenet_loop
from scheduler_frontend import PyScheduler


function_dict = {
    "resnet50": imagenet_loop,
    "resnet101": imagenet_loop,
    "mobilenet_v2": imagenet_loop,
    "bert": None, #bert_loop,
    "gnmt": None, #gnmt_loop,
    "transformer": transformer_loop,
    "retinanet": None, #retinanet_loop,
    "dlrm": None, #dlrm_loop
}

def seed_everything(seed: int):
    import random, os
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def launch_jobs(config_dict_list):

    print(config_dict_list)
    num_clients = len(config_dict_list)
    print(num_clients)

    # init
    start_barriers = [threading.Barrier(2) for i in range(num_clients)]
    end_barriers = [threading.Barrier(2) for i in range(num_clients)]

    print(torch.__version__)

    model_names = [config_dict['arch'] for config_dict in config_dict_list]
    model_files = [config_dict['kernel_file'] for config_dict in config_dict_list]
    num_kernels = [config_dict['num_kernels'] for config_dict in config_dict_list]
    tids = []
    threads = []
    for i, config_dict in enumerate(config_dict_list):
        func = function_dict[config_dict['arch']]
        model_args = config_dict['args']
        model_args.update({"local_rank": 0, "start_barriers": start_barriers, "end_barriers": end_barriers, "tid": i})

        thread = threading.Thread(target=func, kwargs=model_args)
        thread.start()
        tids.append(thread.native_id)
        threads.append(thread)

    print(tids)

    print("before starting")

    timings=[]
    for i in range(10):
        start = time.time()
        print(f"start iter {i}")
        start_barriers[0].wait()
        end_barriers[0].wait()
        torch.cuda.synchronize()
        print(f"Part A took {time.time()-start}")
        # startB = time.time()
        # start_barriers[1].wait()
        # end_barriers[1].wait()
        # torch.cuda.synchronize()
        total_time = time.time()-start
        #print(f"Part B took {time.time()-startB}")
        print(f"Iteration {i} took {total_time} sec")
        timings.append(total_time)

    timings = timings[2:]
    print(f"Avg is {np.median(np.asarray(timings))} sec")

    for thread in threads:
        thread.join()

    print("train joined!")

    print("--------- all threads joined!")

if __name__ == "__main__":
    config_file = sys.argv[1]
    with open(config_file) as f:
        config_dict = json.load(f)
    launch_jobs(config_dict)
