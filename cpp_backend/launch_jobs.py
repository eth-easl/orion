import argparse
import json
import threading
import time
from ctypes import *
import os
import sys
from torchvision import models
import torch

# sys.path.insert(0, "/home/image-varuna/DeepLearningExamples/PyTorch/Translation/GNMT")
# from benchmark_suite.gnmt_trainer import gnmt_loop
sys.path.append("/home/image-varuna/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch")
sys.path.append("/home/image-varuna/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch/utils")
from benchmark_suite.transformer_trainer import transformer_loop
sys.path.append("/home/image-varuna/DeepLearningExamples/PyTorch/LanguageModeling/BERT")
from bert_trainer import bert_loop
sys.path.append("/home/image-varuna/mlcommons/single_stage_detector/ssd")
from benchmark_suite.retinanet_trainer import retinanet_loop
sys.path.append("/home/image-varuna/DeepLearningExamples/PyTorch/Recommendation/DLRM")
#from benchmark_suite.dlrm_trainer import dlrm_loop

#from benchmark_suite.train_imagenet import imagenet_loop
from benchmark_suite.train_imagenet import imagenet_loop
from benchmark_suite.conv_trainer import conv_loop
from benchmark_suite.bnorm_trainer import bnorm_loop
from benchmark_suite.conv_bn_trainer import conv_bn_loop

from scheduler_frontend import PyScheduler

function_dict = {
    "resnet50": imagenet_loop,
    "resnet101": imagenet_loop,
    "mobilenet_v2": imagenet_loop,
    "efficientnet": imagenet_loop,
    "conv": conv_loop,
    "bnorm": bnorm_loop,
    "conv_bnorm": conv_bn_loop,
    "bert": bert_loop,
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

def launch_jobs(config_dict_list, profile, reef_depth, hp_limit, update_start, run_eval):

    seed_everything(42)

    print(config_dict_list)
    num_clients = len(config_dict_list)
    print(num_clients)

    s = torch.cuda.Stream()

    # init

    num_barriers = num_clients+1 if profile else 2
    barriers = [threading.Barrier(num_barriers) for i in range(num_clients)]
    client_barrier = threading.Barrier(num_clients)
    home_directory = os.path.expanduser( '~' )
    if run_eval:
        sched_lib = cdll.LoadLibrary(home_directory + "/gpu_share_repo/cpp_backend/scheduler/scheduler_eval.so")
    else:
        sched_lib = cdll.LoadLibrary(home_directory + "/gpu_share_repo/cpp_backend/scheduler/scheduler.so")
    py_scheduler = PyScheduler(sched_lib, num_clients)

    print(torch.__version__)

    model_names = [config_dict['arch'] for config_dict in config_dict_list]
    model_files = [config_dict['kernel_file'] for config_dict in config_dict_list]

    additional_model_files = [config_dict['additional_kernel_file'] if 'additional_kernel_file' in config_dict else None for config_dict in config_dict_list]
    num_kernels = [config_dict['num_kernels'] for config_dict in config_dict_list]
    num_iters = [config_dict['num_iters'] for config_dict in config_dict_list]
    train_list = [config_dict['args']['train'] for config_dict in config_dict_list]
    additional_num_kernels = [config_dict['additional_num_kernels'] if 'additional_num_kernels' in config_dict else None  for config_dict in config_dict_list]
    tids = []
    threads = []
    for i, config_dict in enumerate(config_dict_list):
        func = function_dict[config_dict['arch']]
        model_args = config_dict['args']
        model_args.update({"num_iters":num_iters[i], "local_rank": 0, "barriers": barriers, "client_barrier": client_barrier, "tid": i})

        thread = threading.Thread(target=func, kwargs=model_args)
        thread.start()
        tids.append(thread.native_id)
        threads.append(thread)

    print(tids)

    sched_thread = threading.Thread(
        target=py_scheduler.run_scheduler,
        args=(
            barriers,
            tids,
            model_names,
            model_files,
            additional_model_files,
            num_kernels,
            additional_num_kernels,
            num_iters,
            profile,
            run_eval,
            False,
            True,
            reef_depth,
            hp_limit,
            update_start,
            train_list
        )
    )

    sched_thread.start()

    for thread in threads:
        thread.join()

    print("train joined!")

    sched_thread.join()
    print("sched joined!")

    print("--------- all threads joined!")

if __name__ == "__main__":
    torch.cuda.set_device(0)
    # affinity_mask = {0,1,2,3}
    # os.sched_setaffinity(0, affinity_mask)
    config_file = sys.argv[1]
    reef_depth = int(sys.argv[2])
    hp_limit = int(sys.argv[3])
    update_start = int(sys.argv[4])
    profile = True
    with open(config_file) as f:
        config_dict = json.load(f)
    launch_jobs(config_dict, profile, reef_depth, hp_limit, update_start, True)
